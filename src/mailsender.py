import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

import os
from os.path import basename

from configuration import config_deeplearing

my_config = config_deeplearing()

# The mail addresses and password
SENDER_ADRESS = my_config["MAILS"]["SENDER_ADRESS"]
SENDER_PASSWORD = my_config["MAILS"]["SENDER_PASSWORD"]

# The outputs path
OUTPUT_PATH = my_config["MAILS"]["OUTPUT_PATH"]

RECEIVER_ADRESS = [
    "challengedeepcs@gmail.com",
    "antoine.pagneux@student-cs.fr",
    "matthieu.briet@student-cs.fr",
    "tanguy.colleville@student-cs.fr",
]


def send_csv_by_mail(
    outputs_path: str = OUTPUT_PATH,
    sender_adress: str = SENDER_ADRESS,
    password: str = SENDER_PASSWORD,
    receiver_adresses: list = RECEIVER_ADRESS,
):
    """[Send the result of a training by mail]

    Args:
        outputs_path (str, optional): [path of the outputs folder]. Defaults to OUTPUT_PATH.
        sender_adress (str, optional): [the mail adress of the sender]. Defaults to SENDER_ADRESS.
        password (str, optional): [the password of the sender]. Defaults to SENDER_PASSWORD.
        receiver_adresses (list, optional): [list of the mail adresses of all desired receivers]. Defaults to RECEIVER_ADRESS.
    """
    files = os.listdir(OUTPUT_PATH)
    for receiver_adress in receiver_adresses:
        # Setup the MIME
        message = MIMEMultipart()
        message["From"] = sender_adress
        message["To"] = receiver_adress
        message["Subject"] = "TRAINING ENDED -> New CSV file"
        # The subject line
        # The body and the attachments for the mail
        mail_content = "Hello young data scientist, please find enclosed the following CSV files : \n"
        for file in files:
            mail_content += " - " + file + "\n"
        mail_content += "Best regards,\nThe developper ;)"
        message.attach(MIMEText(mail_content, "plain"))

        for file in files:
            attach_file_name = outputs_path + file
            attach_file = open(attach_file_name, "rb")  # Open the file as binary mode
            payload = MIMEBase("application", "pdf", Name=file)
            payload.set_payload((attach_file).read())
            encoders.encode_base64(payload)  # encode the attachment
            # add payload header with filename
            payload.add_header(
                "Content-Decomposition", "attachment; filename= %s" % basename(file)
            )
            message.attach(payload)

        # Create SMTP session for sending the mail
        session = smtplib.SMTP("smtp.gmail.com", 587)  # use gmail with port
        session.starttls()  # enable security
        session.login(sender_adress, password)  # login with mail_id and password
        text = message.as_string()
        session.sendmail(sender_adress, receiver_adress, text)
        session.quit()
        print("Mail Sent to : ", receiver_adress)


if __name__ == "__main__":
    print(FILES)
    print(OUTPUT_PATH)
    send_csv_by_mail()
