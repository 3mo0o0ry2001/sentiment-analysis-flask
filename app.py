import spacy
from spacy.language import Language
from transformers import pipeline
from spacy.tokens import Doc

# نجهز محلل المشاعر من مكتبة transformers
sentiment_analyzer = pipeline("sentiment-analysis")

# نضيف مكون spaCy لتصنيف المشاعر
@Language.component("sentiment_component")
def sentiment_component(doc):
    result = sentiment_analyzer(doc.text)[0]
    doc._.sentiment = result
    return doc

# إضافة خاصية sentiment للـ doc
Doc.set_extension("sentiment", default=None)

# تحميل موديل spaCy الأساسي
nlp = spacy.load("en_core_web_sm")

# إضافة مكون تصنيف المشاعر في خط المعالجة
nlp.add_pipe("sentiment_component", last=True)

# نص للاختبار
text = "I love using spaCy! It's so fast and easy. But sometimes NLTK feels like a dinosaur."

doc = nlp(text)

print(f"Text: {doc.text}")
print(f"Sentiment: {doc._.sentiment['label']} (score: {doc._.sentiment['score']:.2f})")
