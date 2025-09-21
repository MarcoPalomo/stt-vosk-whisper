import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import os
import re
from collections import Counter

class TextClassifier:
    def __init__(self):
        """Initialise le classifieur de texte avec LightGBM"""
        self.model = None
        self.vectorizer = None
        self.classes = [
            "réunion", "interview", "cours", "présentation", 
            "discussion", "conférence", "brainstorming", "autres"
        ]
        
        # Mots-clés pour chaque catégorie
        self.keywords = {
            "réunion": ["réunion", "meeting", "ordre du jour", "points", "décision", "action", "suivi"],
            "interview": ["interview", "entretien", "candidat", "poste", "expérience", "compétences", "cv"],
            "cours": ["cours", "leçon", "chapitre", "exercice", "apprendre", "étudier", "examen"],
            "présentation": ["présentation", "slides", "diapo", "présenter", "montrer", "expliquer"],
            "discussion": ["discussion", "débat", "opinion", "avis", "point de vue", "argument"],
            "conférence": ["conférence", "keynote", "speaker", "audience", "public", "présentation"],
            "brainstorming": ["brainstorm", "idées", "créativité", "innovation", "solutions", "propositions"],
        }
        
        self.model_path = "models/text_classifier.pkl"
        self.vectorizer_path = "models/text_vectorizer.pkl"
        
        self._load_or_create_model()
    
    def _generate_synthetic_data(self):
        """Génère des données synthétiques pour l'entraînement"""
        texts = []
        labels = []
        
        synthetic_data = {
            0: [  # meeting
                "Bonjour à tous, nous allons commencer notre réunion hebdomadaire. Premier point à l'ordre du jour.",
                "Il faut prendre une décision sur le budget. Qui peut nous faire un retour sur les chiffres ?",
                "Action à suivre pour la semaine prochaine. Marie, peux-tu t'occuper de contacter les clients ?",
                "Réunion de suivi du projet. Où en sommes-nous avec les livrables ?",
                "Meeting d'équipe pour faire le point sur l'avancement des tâches."
            ],
            1: [  # interview
                "Pouvez-vous me parler de votre expérience professionnelle ? Quelles sont vos compétences principales ?",
                "Entretien d'embauche pour le poste de développeur. Pourquoi voulez-vous rejoindre notre entreprise ?",
                "Interview candidat. Décrivez-moi un projet dont vous êtes particulièrement fier.",
                "Quels sont vos points forts et vos axes d'amélioration ? Parlez-moi de votre CV.",
                "Entretien technique. Comment aborderiez-vous ce type de problème ?"
            ],
            2: [  # training course
                "Aujourd'hui nous allons étudier le chapitre trois. Ouvrez vos livres page quarante-deux.",
                "Cours de mathématiques. Nous allons voir les équations du second degré avec des exercices pratiques.",
                "Leçon sur la Révolution française. Quelles étaient les causes principales de cet événement ?",
                "Apprendre les bases de la programmation. Variables, fonctions et boucles seront au programme.",
                "Examen blanc la semaine prochaine. Révisez bien tous les chapitres que nous avons vus."
            ],
            3: [  # presentation
                "Je vais vous présenter les résultats de notre étude. Comme vous pouvez le voir sur ce graphique.",
                "Présentation du nouveau produit. Voici les fonctionnalités principales que nous avons développées.",
                "Slides suivantes montrent l'évolution du marché. Les tendances sont très encourageantes.",
                "Je vais vous expliquer notre stratégie commerciale pour l'année prochaine avec ces diapos.",
                "Présentation des résultats trimestriels. Nous avons dépassé nos objectifs de quinze pour cent."
            ],
            4: [  # talk
                "Qu'est-ce que vous en pensez ? J'aimerais avoir votre avis sur cette proposition.",
                "Discussion ouverte sur le sujet. Chacun peut donner son point de vue librement.",
                "Débat sur la meilleure approche à adopter. Les arguments des deux côtés sont intéressants.",
                "Opinion personnelle, je pense qu'il faut considérer d'autres options avant de décider.",
                "Échange d'idées constructif. Vos arguments sont pertinents et méritent réflexion."
            ],
            5: [  # conference
                "Mesdames et messieurs, bienvenue à cette conférence sur l'intelligence artificielle.",
                "Le speaker aujourd'hui est un expert reconnu dans son domaine. Applaudissons-le.",
                "Conférence keynote sur les tendances technologiques. L'audience semble très intéressée.",
                "Intervention devant un public de professionnels. Les questions seront les bienvenues.",
                "Présentation magistrale sur les enjeux climatiques. Merci à tous d'être présents."
            ],
            6: [  # brainstorming
                "Séance de brainstorming pour trouver des solutions innovantes au problème.",
                "Toutes les idées sont bonnes à prendre. Laissons libre cours à notre créativité.",
                "Brainstorm collectif. Comment pourrait-on améliorer l'expérience utilisateur ?",
                "Session d'innovation. Proposez toutes vos idées, même les plus farfelues.",
                "Génération d'idées pour le nouveau projet. Pensons différemment et sortons des sentiers battus."
            ],
            7: [  # other
                "Conversation informelle entre collègues. Comment ça va ? Quoi de neuf ?",
                "Discussion générale sans sujet précis. On parle un peu de tout et de rien.",
                "Échange casual sur la météo et les projets du week-end.",
                "Conversation libre sans objectif particulier. Simple échange amical.",
                "Discussion spontanée qui dérive sur plusieurs sujets différents."
            ]
        }
        
        for label, text_list in synthetic_data.items():
            for text in text_list:
                texts.append(text)
                labels.append(label)
                
        return texts, labels
    
    def classify(self, text):
        """
        Classifie un texte et retourne la catégorie avec la confiance
        
        Args:
            text (str): Texte à classifier
            
        Returns:
            dict: {'label': str, 'confidence': float, 'probabilities': dict}
        """
        if not text or len(text.strip()) < 10:
            return {
                'label': 'autres',
                'confidence': 0.5,
                'probabilities': {}
            }
        
        try:
            # Fallback sur règles simples si le modèle n'est pas disponible
            if self.model is None or self.vectorizer is None:
                return self._rule_based_classification(text)
            
            # Vectorisation
            X = self.vectorizer.transform([text])
            
            # Prédiction
            probabilities = self.model.predict(X)[0]
            predicted_class = np.argmax(probabilities)
            confidence = np.max(probabilities)
            
            # Formatage des résultats
            prob_dict = {
                self.classes[i]: float(prob) 
                for i, prob in enumerate(probabilities)
            }
            
            return {
                'label': self.classes[predicted_class],
                'confidence': float(confidence),
                'probabilities': prob_dict
            }
            
        except Exception as e:
            print(f" Erreur lors de la classification: {e}")
            return self._rule_based_classification(text)
    
    def _rule_based_classification(self, text):
        """Classification basée sur des règles simples (fallback)"""
        text_lower = text.lower()
        scores = {}
        
        # Compter les mots-clés pour chaque catégorie
        for category, keywords in self.keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                scores[category] = score
        
        if not scores:
            return {
                'label': 'autres',
                'confidence': 0.6,
                'probabilities': {}
            }
        
        # Trouver la catégorie avec le plus de mots-clés
        best_category = max(scores, key=scores.get)
        confidence = min(0.95, 0.4 + (scores[best_category] * 0.1))
        
        return {
            'label': best_category,
            'confidence': confidence,
            'probabilities': scores
        }

# Instance globale du classifieur
_classifier = None

def get_classifier():
    """Retourne l'instance du classifieur (singleton)"""
    global _classifier
    if _classifier is None:
        _classifier = TextClassifier()
    return _classifier

def classify_text(text):
    """
    Fonction helper pour classifier un texte
    
    Args:
        text (str): Texte à classifier
        
    Returns:
        dict: Résultat de classification
    """
    if not text or len(text.strip()) < 5:
        return {
            'label': 'autres',
            'confidence': 0.5,
            'probabilities': {}
        }
    
    classifier = get_classifier()
    return classifier.classify(text)
    
