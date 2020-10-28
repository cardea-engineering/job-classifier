import yaml


def get_rules(rule_name='job_type'):
    with open("manual_rules/%s_rules.yml" %rule_name, 'r') as stream:
        try:
            data = yaml.safe_load(stream)
            return data
        except yaml.YAMLError as exc:
            print(exc)


job_type_rules = get_rules('job_type')
# job_exp_rules = get_rules('job_exp')


def match_job_type_rules(title, desc=''):
    for role in job_type_rules:
        for keyword in job_type_rules[role]['title']:
            if keyword in title:
                return role
    return None
