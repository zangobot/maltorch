import string

def get_alphanum_chars(s):
    return ''.join(filter(lambda x: x in string.ascii_letters + string.digits + string.punctuation, s))
