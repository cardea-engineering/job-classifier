import yaml


def get_rules():
    with open("static/manual_rules.yml", 'r') as stream:
        try:
            data = yaml.safe_load(stream)
            return data
        except yaml.YAMLError as exc:
            print(exc)


rules = get_rules()


def get_match(title, desc):
    for role in rules:
        for keyword in rules[role]['title']:
            if keyword in title:
                return role
    return False
