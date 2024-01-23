import threading
from django.contrib.auth.tokens import PasswordResetTokenGenerator
from six import text_type
from django.core.mail import EmailMessage

class EmailThread(threading.Thread):
    def __init__(self, email):
        self.email = email
        threading.Thread.__init__(self)

    def run(self):
        self.email.send(fail_silently=True)

class AppTokenGenerator(PasswordResetTokenGenerator):
    def _make_hash_value(self, user, timestamp):
        return (text_type(user.is_active)+text_type(user.pk)+text_type(timestamp))

token_generater = AppTokenGenerator()

def extract_first_last_name(full_name):
    if not full_name:
        return "", ""
    parts = full_name.split()
    first_name = parts[0]
    last_name = ' '.join(parts[1:]) if len(parts) > 1 else ''
    return first_name, last_name

def SendEmail(email_subject,email_body,from_mail,email):
    email = EmailMessage(
        email_subject,
        email_body,
        from_mail,
        [email],
    )
    email.content_subtype = 'html'
    EmailThread(email).start()