import numpy as np
from flask import Flask, request, jsonify
from utils import *


job_category_model = get_serialized('job_type_category_name_model.pkl')
job_type_model = get_serialized('job_type_name_model.pkl')
tfidfVectorizer = get_serialized('tfIdfVectorizer.pkl')

job_categories = ['Business Operations', 'Data & Analytics',
                  'Finance, Legal & Compliance', 'Product & Design',
                  'Software Engineering']

job_types = ['Back-End Software Engineering', 'Business Development',
             'Business Intelligence & Data Analysis', 'Data Engineering',
             'DevOps & Infrastructure', 'Front-End Software Engineering',
             'Full-Stack Software Engineering', 'Operations & General Business',
             'Product Manager', 'Sales']

app = Flask(__name__)


def wrap_result(names, probs):
    return sorted([
        {'name': names[i], 'probability': float('%.3f' % p)}
        for i, p in enumerate(probs)], key=lambda x: -x['probability']
    )


@app.route('/predict', methods=['POST'])
def results():

    data = request.form
    job_title = data['title']
    job_desc = data['desc']
    text_input = parse_raw_html(job_title + job_desc)

    Xts = tfidfVectorizer.transform(np.array([text_input])).toarray()

    prob_category = job_category_model.predict_proba(Xts).flatten()
    prob_type = job_type_model.predict_proba(Xts).flatten()

    result_category = wrap_result(job_categories, prob_category)
    result_type = wrap_result(job_types, prob_type)

    return jsonify({
        'job_category': result_category,
        'job_type': result_type
    })


if __name__ == "__main__":
    app.run(debug=True)  # for development
    # app.run(debug=False)  # for production
