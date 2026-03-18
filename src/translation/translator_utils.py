class EnHiTranslator:
    def __init__(self):
        self.full_map = {
            "please take your medicines on time.": "कृपया अपनी दवाइयाँ समय पर लें।",
            "seek urgent care if chest pain worsens.": "यदि छाती का दर्द बढ़े तो तुरंत चिकित्सा लें।",
            "drink enough water to avoid dehydration.": "डिहाइड्रेशन से बचने के liye पर्याप्त पानी पिएँ।",
            "what should i do if chest pain worsens?": "यदि छाती का दर्द बढ़े तो मुझे क्या करना चाहिए?",
            "the doctor was very helpful.": "डॉक्टर बहुत सहायक थे।",
            "the nursing staff were caring and attentive.": "नर्सिंग स्टाफ देखभाल करने वाला और सतर्क था।",
            "long waiting time and poor coordination.": "लंबा प्रतीक्षा समय और खराब समन्वय।",
            "please consult your doctor.": "कृपया अपने डॉक्टर से परामर्श करें।",
            "take rest and drink water.": "आराम करें और पानी पिएँ।",
            "follow the discharge instructions carefully.": "डिस्चार्ज निर्देशों का सावधानीपूर्वक पालन करें।"
        }

        self.word_map = {
            "doctor": "डॉक्टर",
            "nurse": "नर्स",
            "nursing": "नर्सिंग",
            "staff": "स्टाफ",
            "patient": "रोगी",
            "hospital": "अस्पताल",
            "medicine": "दवा",
            "medicines": "दवाइयाँ",
            "water": "पानी",
            "pain": "दर्द",
            "chest": "छाती",
            "fever": "बुखार",
            "care": "देखभाल",
            "urgent": "तुरंत",
            "drink": "पिएँ",
            "take": "लें",
            "please": "कृपया",
            "time": "समय",
            "helpful": "सहायक",
            "clean": "साफ",
            "service": "सेवा",
            "delay": "देरी",
            "delayed": "विलंबित",
            "discharge": "डिस्चार्ज",
            "instructions": "निर्देश",
            "rest": "आराम",
            "consult": "परामर्श करें",
            "your": "अपना",
            "on": "पर",
            "if": "यदि",
            "worsens": "बढ़े",
            "avoid": "बचें",
            "dehydration": "डिहाइड्रेशन"
        }

    def translate(self, text: str, max_length: int = 128) -> str:
        if not text:
            return ""

        s = text.strip().lower()
        if s in self.full_map:
            return self.full_map[s]

        tokens = s.replace("?", " ?").replace(".", " .").split()
        out = []
        for tok in tokens:
            out.append(self.word_map.get(tok, tok))

        translated = " ".join(out)
        translated = translated.replace(" .", "।").replace(" ?", "?")
        return translated
