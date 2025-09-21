from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class TextSummarizer:
    def __init__(self):
        """Initialise le résumeur avec T5 en français"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔄 Chargement du modèle de résumé sur {self.device}...")
        
        # Utiliser T5 pour le français
        model_name = "csebuetnlp/mT5_multilingual_XLSum"
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.model.to(self.device)
            
            # Pipeline de résumé
            self.summarizer = pipeline(
                "summarization",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )
            print("✅ Modèle de résumé chargé avec succès !")
            
        except Exception as e:
            print(f"❌ Erreur chargement modèle de résumé: {e}")
            # Fallback sur un modèle plus simple
            self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    
    def summarize(self, text, max_length=150, min_length=50):
        """
        Résume le texte donné
        
        Args:
            text (str): Texte à résumer
            max_length (int): Longueur maximale du résumé
            min_length (int): Longueur minimale du résumé
            
        Returns:
            str: Texte résumé
        """
        if not text or len(text.strip()) < 50:
            return "Texte trop court pour être résumé."
        
        try:
            # Préfixer pour T5 multilingue
            if "t5" in str(self.model.__class__).lower():
                input_text = f"summarize: {text}"
            else:
                input_text = text
            
            # Résumé
            summary = self.summarizer(
                input_text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
                truncation=True
            )
            
            return summary[0]['summary_text']
            
        except Exception as e:
            print(f"❌ Erreur lors du résumé: {e}")
            # Résumé de fallback simple
            return self._simple_summary(text)
    
    def _simple_summary(self, text, max_sentences=3):
        """Résumé simple en prenant les premières phrases"""
        sentences = text.split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= max_sentences:
            return text
        
        return '. '.join(sentences[:max_sentences]) + '.'

# Instance globale du résumeur
_summarizer = None

def get_summarizer():
    """Retourne l'instance du résumeur (singleton)"""
    global _summarizer
    if _summarizer is None:
        _summarizer = TextSummarizer()
    return _summarizer

def summarize_text(text, max_length=150, min_length=50):
    """
    Fonction helper pour résumer un texte
    
    Args:
        text (str): Texte à résumer
        max_length (int): Longueur max du résumé
        min_length (int): Longueur min du résumé
        
    Returns:
        str: Résumé du texte
    """
    if not text or len(text.strip()) < 20:
        return "Pas assez de contenu pour générer un résumé."
    
    summarizer = get_summarizer()
    return summarizer.summarize(text, max_length, min_length)