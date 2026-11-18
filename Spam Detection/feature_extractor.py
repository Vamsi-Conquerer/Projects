import os, glob, re, math, time
from collections import Counter
import pandas as pd
import email
from email import policy
from email.parser import BytesParser
from urllib.parse import urlparse
import numpy as np

# -------------------------
# FIXED PARSER (ONLY CHANGE)
# -------------------------
def safe_parse_email(path):
    try:
        with open(path, 'rb') as fp:
            raw = fp.read()

        # Try strict parsing
        try:
            return BytesParser(policy=policy.default).parsebytes(raw)
        except:
            pass

        # Fallback for Enron-style/raw emails
        try:
            txt = raw.decode('utf-8', errors='ignore')
            return email.message_from_string(txt, policy=policy.default)
        except:
            return None

    except Exception as e:
        print(f"❌ Failed: {path}")
        return None


# -------------------------
# ORIGINAL HELPERS (UNCHANGED)
# -------------------------
def get_header(msg, name):
    try: return msg.get(name)
    except: return None

def domain_of(address):
    if not address: return ''
    m = re.search(r'@([A-Za-z0-9.-]+)', str(address))
    return m.group(1).lower() if m else ''

def safe_body_text(msg):
    try:
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == 'text/plain':
                    return part.get_content()
        else:
            return msg.get_payload()
    except:
        return ''
    return ''


# -------------------------
# ORIGINAL FEATURE FUNCTION (UNCHANGED)
# -------------------------
def extract_features_from_msg(path):
    msg = safe_parse_email(path)
    if msg is None: return None

    feat = {}

    feat['message_id'] = get_header(msg, 'Message-ID') or ''
    feat['from'] = get_header(msg, 'From') or ''
    feat['from_domain'] = domain_of(feat['from'])
    feat['return_path'] = get_header(msg, 'Return-Path') or ''
    feat['return_path_domain'] = domain_of(feat['return_path'])

    feat['num_recipients'] = len(msg.get_all('To', []) or []) + len(msg.get_all('Cc', []) or [])

    raw_date = get_header(msg, 'Date') or ''
    try:
        dt = email.utils.parsedate_to_datetime(raw_date) if raw_date else None
        feat['time_since_epoch_days'] = int(dt.timestamp() // 86400) if dt else -1
        feat['hour_of_day'] = dt.hour if dt else -1
        feat['weekday'] = dt.weekday() if dt else -1
    except:
        feat['time_since_epoch_days'] = -1
        feat['hour_of_day'] = -1
        feat['weekday'] = -1

    recv = msg.get_all('Received', []) or []
    feat['received_hops'] = len(recv)
    feat['has_list_unsubscribe'] = int(bool(get_header(msg, 'List-Unsubscribe')))

    subj = get_header(msg, 'Subject') or ''
    body = safe_body_text(msg) or ''
    feat['subject_raw'] = subj
    feat['body_raw'] = body

    text = re.sub(r'\r\n', ' ', body)
    words = re.findall(r"[A-Za-z0-9']+", text)

    feat['word_count'] = len(words)
    feat['char_count'] = len(text)

    feat['subject_exclamation_count'] = subj.count('!')
    feat['subject_all_caps_words'] = sum(1 for w in subj.split() if w.isupper())
    feat['urgent_subject'] = int(bool(re.search(r'\b(urgent|important)\b', subj, re.I)))

    feat['num_urls'] = len(re.findall(r'https?://[^\s<>"]+', text))
    feat['ip_url_count'] = len([u for u in re.findall(r'https?://[^\s<>"]+', text) if re.search(r'://\d+\.\d+\.\d+\.\d+', u)])
    feat['is_html'] = int(bool(re.search(r'<[^>]+>', body)))

    attach_cnt = 0
    exec_attach = 0
    try:
        for part in msg.walk():
            if part.get_content_maintype() == 'multipart': continue
            fname = part.get_filename()
            if fname:
                attach_cnt += 1
                if fname.lower().endswith(('.exe', '.bat', '.scr', '.vbs')):
                    exec_attach += 1
    except:
        pass

    feat['attachment_count'] = attach_cnt
    feat['attachments_with_executable'] = exec_attach

    return feat


# -------------------------
# FIXED PROCESS (ONLY DEBUG ADDED)
# -------------------------
def process_dir(folder, label):
    results = []
    files = glob.glob(os.path.join(folder, '*'))

    print(f"\n📂 {label}: {len(files)} files found")

    for i, fpath in enumerate(files, start=1):
        feat = extract_features_from_msg(fpath)

        if feat is None:
            print(f"⚠️ Skipped: {fpath}")
            continue

        feat['label'] = label
        feat['filename'] = os.path.basename(fpath)
        results.append(feat)

        if i % 500 == 0:
            print(f"Processed {i} in {label}...")

    return results


# -------------------------
# MAIN (UNCHANGED)
# -------------------------
if __name__ == "__main__":

    base = {
        'spam': r"D:\Main Project\Main Project\Dataset_4_categories\spam",
        'ham': r"D:\Main Project\Main Project\Dataset_4_categories\ham",
        'promotional': r"D:\Main Project\Main Project\Dataset_4_categories\promotional",
        'malware': r"D:\Main Project\Main Project\Dataset_4_categories\malware"
    }

    all_records = []

    for label, folder in base.items():
        if os.path.exists(folder):
            print(f"\n🚀 Processing {label}...")
            all_records.extend(process_dir(folder, label))
        else:
            print(f"❌ Folder missing: {folder}")

    df = pd.DataFrame(all_records)

    print("\n📊 Total emails processed:", len(df))

    out_dir = r"D:\Main Project\Main Project"
    os.makedirs(out_dir, exist_ok=True)

    outpath = os.path.join(out_dir, "emails_features_4_classes.pkl")
    df.to_pickle(outpath)

    print(f"\n✅ Saved to: {outpath}")