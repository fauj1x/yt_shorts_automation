import spacy

try:
    nlp = spacy.load("id_core_news_sm")
except OSError:
    raise RuntimeError(
        "Model SpaCy 'id_core_news_sm' belum terpasang. Install wheel-nya dari Hugging Face:\n"
        "https://huggingface.co/firqaaa/id_core_news_sm"
    )


# Tambahkan sentencizer jika belum ada (untuk deteksi kalimat)
if "sentencizer" not in nlp.pipe_names:
    nlp.add_pipe("sentencizer")

def analyze_text(text: str) -> dict:
    """
    Analisis teks transkrip dan hasilkan keyword & judul otomatis.
    """
    doc = nlp(text)

    # Deteksi kalimat
    sentences = list(doc.sents)
    title = sentences[0].text.strip() if sentences else "Video Pendek"

    # Deteksi entitas (NER)
    entities = {ent.text.lower() for ent in doc.ents}

    # Keyword dari lemma token yang valid
    keywords = {
        token.lemma_.lower()
        for token in doc
        if not token.is_stop and token.is_alpha and len(token) > 2
    }

    return {
        "title": title,
        "keywords": sorted(keywords.union(entities))
    }
