import re

class String:
    def __init__(self, text):
        self.text = text
    
    def cap_format(self):
        data_filter = r"[\.\_/]+"
        precleaned = re.sub(data_filter, "", self.text)
        cleaned = " ".join([k.capitalize() for k in re.findall(r"\w+", precleaned)])
        return cleaned
        