# 
# In Windows terminal: Run 
# c:\> chcp 65001 
# to set character encoding

# Patch to lib/email/header.py:decode_header:79
# if hasattr(header, '_chunks'):
#     return [(_charset._encode(string, str(charset)), str(charset))
#                 for string, charset in header._chunks]
# # If no encoding, just return the header with no charset.
# if not ecre.search(header):
#     return [(header, None)]
# ---
# if not header or not ecre.search(header):
#     return [(header, None)]
#
# @todo Refactor to caller in imapy

import codecs
import datetime
import getpass
import io
import sys

import imapy
from imapy.query_builder import Q
from html.parser import HTMLParser
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import yaml

import locale
locale.setlocale(locale.LC_ALL, '')

# Needs os.env['PYTHONIOENCODING'] == UTF-8. 


if sys.argv[1] not in ['train', 'sort']:
    sys.stderr.write('Valid subcommands: train, sort\n')
    sys.exit(2)
else:
    subcommand = sys.argv[1]

# Config file of the form:
#
# hostname: imap.email.com
# port: 993
# username: jane.q.public@email.com
#
# You will be prompted for the password
#
    
config_filename = 'my_email.yaml'
with open(config_filename) as f:
    config = yaml.load(f)
hostname = config['hostname']
port = config['port']
username = config['username']
password = getpass.getpass()

    
# Makes output default to unicode.
#
# Thanks Craig McQueen:
# http://stackoverflow.com/questions/492483/setting-the-correct-encoding-when-piping-stdout-in-python
# sys.stdout = codecs.getwriter('utf8')(sys.stdout)

# Thanks eryksun
# sys.stdout = io.TextIOWrapper(sys.stdout.detach(), sys.stdout.encoding, 'replace')


inbox = imapy.connect(host='imap.sri.com', port=port, username=username, password=password, ssl=True)
excluded = ['INBOX/Notifications']
folder_names = [f for f in inbox.folders() if 'INBOX/' in f and f not in excluded]
#query_no_huge = Q().smaller("1 MB")  # Otherwise downloading lots of emails is very slow
query_no_huge = Q().smaller("25 KB")

# Thanks Ooker:
# http://stackoverflow.com/questions/753052/strip-html-from-strings-in-python
class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)

rm_html = MLStripper()
def strip_html(html):
    rm_html.feed(html)
    ret = rm_html.get_data()
    rm_html.reset()
    return ret
    
def body_as_plain_text(email):
    if email['text']:
        html = email['text'][0]['text']
        return strip_html(html)
    elif email['html']:
        html = email['html'][0]
        return strip_html(html)
    else:
        return ''

if subcommand == 'train':
    # Generate training set, format and shape for sklearn
    target_names = folder_names
    target = []
    ids = []
    subjects = []
    bodies = []
  
    print("Downloading messages by folder...")
    for i_folder, folder_name in enumerate(folder_names):
        print(folder_name)
        try:
            folder = inbox.folder(folder_name)
            emails = folder.emails(query_no_huge)
            for email in emails:
                id = email['headers']['message-id'][0].strip()
                subject = email['subject']  
                text = body_as_plain_text(email)
                
                ids.append(id)
                subjects.append(email['subject'])
                bodies.append(text)
                target.append(i_folder)
            
        except Exception as e:
            sys.stdout.write('Error: {}\n'.format(e.__str__()))
            continue
            
    print("Counting words...")
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(bodies)
    
    print("Calculating frequencies...", flush=True)
    tfidf_transformer = TfidfTransformer(use_idf=True).fit(X_train_counts)
    X_train_tfidf = tfidf_transformer.transform(X_train_counts)
    
    cross_validate = True
    if cross_validate:
        from sklearn.cross_validation import train_test_split
        _X_train_tfidf = X_train_tfidf
        X_train_tfidf, X_test, target, y_test = train_test_split(
            _X_train_tfidf, target, test_size=0.33, random_state=42)
    
    print("Training...")
    clf = MultinomialNB(alpha=.01)
    clf.fit(X_train_tfidf, target)
    
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')  
    filename = 'email_classifier_{}.pkl'.format(timestamp)
    joblib.dump(clf, filename)
    print('Saved as "{}"'.format(filename))
    
    if cross_validate:
        from sklearn import metrics
        
        predicted = clf.predict(X_test)
        print(
            metrics.classification_report(y_test, predicted, target_names=target_names)
        )
        cm = metrics.confusion_matrix(y_test, predicted)
        # print(cm)
        
        # From http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#example-model-selection-plot-confusion-matrix-py
        import matplotlib.pyplot as plt
        import numpy as np
        def plot_confusion_matrix(cm, target_names='', title='Confusion matrix', cmap=plt.cm.Blues):
            plt.imshow(cm, interpolation='nearest', cmap=cmap)
            plt.title(title)
            plt.colorbar()
            tick_marks = np.arange(len(target_names))
            if target_names:
                plt.xticks(tick_marks, target_names, rotation=45)
                plt.yticks(tick_marks, target_names)
            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            
        plot_confusion_matrix(cm, target_names=target_names)
        plt.show()

        # cls.predict_proba() - probability by class
        # Update training with cls.partial_fit()
        
elif subcommand == 'sort':
    if len(argv) < 3:
        print('Need a training data file.')
        sys.exit(2)
    else:
        training_file = argv[2]
        clf = joblib.load(training_file)

    # Gather data from inbox
    inbox_root = inbox.folder('INBOX')

    ids = []
    subjects = []
    bodies = []
    
    emails = inbox_root.emails(query_no_huge)
    for i_email, email in enumerate(emails):
        try:
            id = email['headers']['message-id'][0].strip()
            subject = email['subject']  
            text = body_as_plain_text(email)
            if not text:
                print('Blank email: {}'.format(subject))
                continue
        except Exception as e:
            print('{}: {} with email #{}: {}'.format(type(e).__name__, e.__str__(), i_email, subject))
            continue
            
        print('#{}: {}'.format(i_email, subject))
            
        ids.append(id)
        subjects.append(subject)
        bodies.append(text)
        
        
        # Retrieve a specific email
        # inbox_root.emails(Q().header('message-id', ids[4]))
        
        # Sort it
        # cls_name = cls_names[cls]
        # print('Moved "{}" to "{}"'.format(subject, cls_name))
        # email.move(cls_name).mark(['Unseen'])
 

# clf.predict(inbox)
    
#f_admin = inbox.folder('INBOX/Admin')
#
#
#for em in f_admin.emails(-2): # Gets most recent
#    text = em['text'][0]['text']
#    rm_html.feed(text)
#    print(rm_html.get_data())
#    print('\n\n-----------------------------\n\n')
#    rm_html.reset()


    
# When run:
# for each new email in Inbox, classify it (move to folder, do NOT mark as read)
# 
# Update training:
# - Find read emails that haven't been used to train yet
# - Append or retrain
