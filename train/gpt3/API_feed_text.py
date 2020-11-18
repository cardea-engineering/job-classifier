

# import sys
# sys.path.append('/Users/gdkou/Projects/Cardea/job_classifier')

# from utils.utils import *


def get_row(title, desc, type):
    return 'Description: "%s: %s"\nType: "%s"\n###\n' % (title, desc, type)


df_list = df.head()[['title','description', 'job_type_name']].values.tolist()


with open("api_feed.txt", 'w') as f:
    f.writelines(['This is a job type classifier\n'])

    for title, desc, type in df_list:
        f.writelines([get_row(title, get_text_from_html(desc), type)])
