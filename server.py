import numpy as np
from flask import Flask, request, jsonify
from utils.utils import *
from utils.match_rules import get_match


job_category_model = get_serialized('job_type_category_name_model.pkl')
job_type_model = get_serialized('job_type_name_model.pkl')
tfidfVectorizer = get_serialized('tfIdfVectorizer.pkl')


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

    type_matched = get_match(job_title, job_desc)
    if type_matched:
        result_type = {type_matched: 1}
    else:
        prob_types = job_type_model.predict_proba(Xts).flatten()
        result_type = wrap_result(job_types, prob_types)

    prob_category = job_category_model.predict_proba(Xts).flatten()
    result_category = wrap_result(job_categories, prob_category)

    return jsonify({
        'job_category': result_category,
        'job_type': result_type
    })


if __name__ == "__main__":
    app.run(debug=True)  # for development
    # app.run(debug=False)  # for production
