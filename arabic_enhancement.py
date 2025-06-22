#!/usr/bin/env python3
# تحسينات النطق العربي - للحساب المجاني

import re
import requests
import tempfile
import subprocess
import platform
import threading
import time
from loguru import logger

# قاموس شامل لتصحيح النطق
ARABIC_PRONUNCIATION_DICT = {
    # مشكلة المحطات الأساسية
    "لاعربي": "سَابِك",
    "سبك": "سَابِك", 
    "لاعربى": "سَابِك",
    "SABIC": "سَابِك",
    "sabic": "سَابِك",
    
    # أسماء المحطات الشائعة
    "الملك عبدالله": "الْمَلِكْ عَبْدُالله",
    "الملك فهد": "الْمَلِكْ فَهْد", 
    "الأمير محمد": "الأَمِيرْ مُحَمَّد",
    "الرياض": "الرِّيَاضْ",
    "المطار": "الْمَطَارْ",
    "الدرعية": "الدِّرْعِيَّة",
    "النور": "النُّورْ",
    "الفيصلية": "الْفَيْصَلِيَّة",
    "المربع": "الْمُرَبَّع",
    "الحكومة": "الْحُكُومَة",
    "الجامعة": "الْجَامِعَة",
    "المستشفى": "الْمُسْتَشْفَى",
    
    # كلمات النظام - مع تحسين النبرة
    "محطة": "مَحَطَّة",
    "مطعم": "مَطْعَم", 
    "مطاعم": "مَطَاعِم",
    "سراج": "سِرَاج",
    "تفضل": "تَفَضَّل",
    "مرحبا": "مَرْحَباً",
    "مرحباً": "مَرْحَباً",
    "اهلا": "أَهْلاً",
    "أهلا": "أَهْلاً وَسَهْلاً",
    "اقرب": "أَقْرَب",
    "مسار": "مَسَار",
    "طريق": "طَرِيق",
    "خريطة": "خَرِيطَة",
    "اضغط": "اضْغَط",
    "عرض": "عَرْض",
    "شكرا": "شُكْراً",
    "شكرًا": "شُكْراً لَك",
    "عفوا": "عَفْواً",
    "عفواً": "عَفْواً",
    
    # عبارات مهذبة ولطيفة
    "ما اسم": "مَا اسْمُ",
    "كيف يمكنني": "كَيْفَ يُمْكِنُني",
    "يسعدني": "يُسْعِدُني",
    "بكل سرور": "بِكُلِّ سُرُور",
    "لا مشكلة": "لا مُشْكِلَة",
    
    # مطاعم شائعة
    "البيك": "الْبِيك",
    "الطازج": "الطَّازِج",
    "كودو": "كُودُو",
    "الشرفة": "الشُّرْفَة",
    "النافورة": "النَّافُورَة",
    "الشيف": "الشِّيف",
    "ستاربكس": "سْتَارْبَكْس",
    "ماكدونالدز": "مَاكْدُونَالْدز",
    "كنتاكي": "كِنْتَاكِي",
    "بيتزا هت": "بِيتْزَا هَت",
    "دومينوز": "دُومِينُوز",
    
    # أرقام وأعداد
    "واحد": "وَاحِد",
    "اثنين": "اثْنَيْن", 
    "ثلاثة": "ثَلَاثَة",
    "اربعة": "أَرْبَعَة",
    "خمسة": "خَمْسَة",
    "ستة": "سِتَّة",
    "سبعة": "سَبْعَة",
    "ثمانية": "ثَمَانِيَة",
    "تسعة": "تِسْعَة",
    "عشرة": "عَشْرَة",
    
    # كلمات مساعدة
    "هذا": "هَذَا",
    "هذه": "هَذِهِ", 
    "ذلك": "ذَلِك",
    "تلك": "تِلْك",
    "الى": "إِلَى",
    "إلى": "إِلَى",
    "من": "مِن",
    "في": "فِي",
    "على": "عَلَى",
    "عند": "عِنْد",
    "بعد": "بَعْد",
    "قبل": "قَبْل",
    "فوق": "فَوْق",
    "تحت": "تَحْت",
    "يمين": "يَمِين",
    "يسار": "يَسَار",
    "شمال": "شِمَال",
    "جنوب": "جَنُوب",
    "شرق": "شَرْق",
    "غرب": "غَرْب",
}

# قواعد إضافة الحركات التلقائية
DIACRITIC_RULES = [
    # أدوات التعريف
    (r'\bال([بتثجحخدذرزسشصضطظعغفقكلمنهوي])', r'الْ\1'),
    (r'\bال([أإ])', r'الْ\1'),
    
    # حروف الجر
    (r'\bمن\s+', r'مِن '),
    (r'\bإلى\s+', r'إِلَى '),
    (r'\bفي\s+', r'فِي '),
    (r'\bعلى\s+', r'عَلَى '),
    (r'\bعن\s+', r'عَن '),
    
    # ضمائر
    (r'\bهو\b', r'هُوَ'),
    (r'\bهي\b', r'هِيَ'),
    (r'\bأنت\b', r'أَنْت'),
    (r'\bأنتم\b', r'أَنْتُم'),
    (r'\bنحن\b', r'نَحْن'),
    
    # أفعال شائعة مع نبرة ودودة
    (r'\bيكون\b', r'يَكُون'),
    (r'\bكان\b', r'كَان'),
    (r'\bيريد\b', r'يُرِيد'),
    (r'\bأريد\b', r'أُرِيد'),
    (r'\bاذهب\b', r'اذْهَب'),
    (r'\bتعال\b', r'تَعَال'),
    (r'\bانتظر\b', r'انْتَظِر'),
]

def enhance_arabic_pronunciation(text: str) -> str:
    """
    Enhance Arabic text for better pronunciation with TTS systems
    """
    # Common pronunciation fixes
    replacements = {
        # Station names
        'لاعربي': 'لا عربي',
        'سابك': 'سَابِك',
        'الملك عبدالله': 'الْمَلِك عَبْد اللَّه',
        'الملك فهد': 'الْمَلِك فَهْد',
        'الملك خالد': 'الْمَلِك خَالِد',
        
        # Restaurant names
        'البيك': 'الْبِيك',
        'كودو': 'كُودُو',
        'هرفي': 'هَرْفِي',
        'ماكدونالدز': 'مَاكْدُونَالْدز',
        'كنتاكي': 'كِنْتَاكِي',
        'بيتزا هت': 'بِيتْزَا هَت',
        'دومينوز': 'دُومِينُوز',
        'صب واي': 'صَب وَاي',
        'برجر كنج': 'بُرْجَر كِنْج',
        'الطازج': 'الطَّازِج',
        
        # Common words
        'مطعم': 'مَطْعَم',
        'محطة': 'مَحَطَّة',
        'مسار': 'مَسَار',
        'أقرب': 'أَقْرَب',
        'إلى': 'إِلَى',
        'من': 'مِن',
        'الرياض': 'الرِّيَاض',
        'المترو': 'الْمِتْرُو',
        
        # Directions and instructions
        'اضغط': 'اضْغَط',
        'على': 'عَلَى',
        'إظهار': 'إِظْهَار',
        'الخريطة': 'الْخَرِيطَة',
        'لعرض': 'لِعَرْض',
        'تفضل': 'تَفَضَّل',
        'ما اسم': 'مَا اسْم',
        'من فضلك': 'مِن فَضْلِك',
        
        # Greetings and responses
        'مرحبا': 'مَرْحَبَاً',
        'أهلا': 'أَهْلاً',
        'وسهلا': 'وَسَهْلاً',
        'شكرا': 'شُكْرَاً',
        'عفوا': 'عَفْوَاً',
        'نعم': 'نَعَم',
        'لا': 'لَا',
        
        # System responses
        'جاهز': 'جَاهِز',
        'للاستماع': 'لِلاِسْتِمَاع',
        'حاول': 'حَاوِل',
        'مرة أخرى': 'مَرَّة أُخْرَى',
        'لم أسمع': 'لَم أَسْمَع',
        'شيئا': 'شَيْئَاً',
        'لم أفهم': 'لَم أَفْهَم',
        'طلبك': 'طَلَبَك',
        'لم أعرف': 'لَم أَعْرِف',
        'اسم المطعم': 'اسْم الْمَطْعَم',
        
        # Numbers (commonly mispronounced)
        'واحد': 'وَاحِد',
        'اثنان': 'اثْنَان',
        'ثلاثة': 'ثَلَاثَة',
        'أربعة': 'أَرْبَعَة',
        'خمسة': 'خَمْسَة',
        'ستة': 'سِتَّة',
        'سبعة': 'سَبْعَة',
        'ثمانية': 'ثَمَانِيَة',
        'تسعة': 'تِسْعَة',
        'عشرة': 'عَشَرَة'
    }
    
    enhanced = text
    
    # Apply replacements
    for original, enhanced_version in replacements.items():
        enhanced = enhanced.replace(original, enhanced_version)
    
    # Add pauses for better flow
    enhanced = re.sub(r'([.!?])', r'\1 ', enhanced)
    enhanced = re.sub(r'،', '، ', enhanced)
    
    # Clean up extra spaces
    enhanced = re.sub(r'\s+', ' ', enhanced).strip()
    
    return enhanced

def remove_diacritics(text: str) -> str:
    """
    Remove Arabic diacritics for cleaner display
    """
    diacritics = (
        '\u064B'  # Fathatan
        '\u064C'  # Dammatan
        '\u064D'  # Kasratan
        '\u064E'  # Fatha
        '\u064F'  # Damma
        '\u0650'  # Kasra
        '\u0651'  # Shadda
        '\u0652'  # Sukun
        '\u0653'  # Maddah
        '\u0654'  # Hamza above
        '\u0655'  # Hamza below
        '\u0656'  # Subscript alef
        '\u0657'  # Inverted damma
        '\u0658'  # Mark noon ghunna
        '\u0659'  # Zwarakay
        '\u065A'  # Vowel sign small v
        '\u065B'  # Vowel sign inverted small v
        '\u065C'  # Vowel sign dot below
        '\u065D'  # Reversed damma
        '\u065E'  # Fatha with two dots
        '\u065F'  # Wavy hamza below
    )
    
    return ''.join(char for char in text if char not in diacritics)

def create_optimized_voice_settings():
    """
    Create optimized voice settings for ElevenLabs Arabic TTS
    """
    return {
        "stability": 0.75,        # Good balance for Arabic
        "similarity_boost": 0.85, # High similarity for consistency
        "style": 0.2,            # Moderate style for natural speech
        "use_speaker_boost": True # Enhance speaker characteristics
    }

def normalize_restaurant_name(name: str) -> str:
    """
    Normalize restaurant names for better matching
    """
    # Remove common prefixes
    name = re.sub(r'^(مطعم|مقهى|كافيه|كافية)\s*', '', name, flags=re.IGNORECASE)
    
    # Normalize spacing
    name = re.sub(r'\s+', ' ', name).strip()
    
    # Common name variations
    variations = {
        'البيك': ['بيك', 'الباك', 'باك'],
        'كودو': ['كودوو', 'كدو'],
        'هرفي': ['هرفى', 'هارفي'],
        'ماكدونالدز': ['ماك', 'مكدونالدز', 'ماكدونالد'],
        'كنتاكي': ['كنتكي', 'كنتاكى'],
        'بيتزا هت': ['بيتزا', 'بيتزاهت'],
        'الطازج': ['طازج']
    }
    
    # Check for variations
    name_lower = name.lower()
    for standard, variants in variations.items():
        if name_lower in [v.lower() for v in variants]:
            return standard
    
    return name

def extract_restaurant_patterns():
    """
    Return comprehensive patterns for restaurant name extraction
    """
    return [
        # Direct restaurant mention
        r"مطعم\s*([\u0600-\u06FF0-9'\-\s]+)",
        
        # Popular restaurants (exact matches)
        r"(البيك|الطازج|كودو|هرفي|ماكدونالدز|كنتاكي|بيتزا\s*هت|دومينوز|صب\s*واي|برجر\s*كنج)",
        
        # Intent-based patterns
        r"أريد\s*(?:أن\s*أذهب\s*إلى\s*)?([\u0600-\u06FF0-9'\-\s]+)",
        r"إلى\s*([\u0600-\u06FF0-9'\-\s]+)",
        r"عند\s*([\u0600-\u06FF0-9'\-\s]+)",
        r"في\s*([\u0600-\u06FF0-9'\-\s]+)",
        
        # General Arabic text (fallback)
        r"([\u0600-\u06FF]{3,}(?:\s+[\u0600-\u06FF]+)*)"
    ]

def get_restaurant_shortcuts():
    """
    Return dictionary of restaurant shortcuts for quick matching
    """
    return {
        'البيك': 'البيك',
        'بيك': 'البيك',
        'كودو': 'كودو',
        'هرفي': 'هرفي',
        'الطازج': 'الطازج',
        'طازج': 'الطازج',
        'ماكدونالدز': 'ماكدونالدز',
        'ماك': 'ماكدونالدز',
        'مكدونالدز': 'ماكدونالدز',
        'كنتاكي': 'كنتاكي',
        'كنتكي': 'كنتاكي',
        'بيتزا': 'بيتزا هت',
        'بيتزاهت': 'بيتزا هت',
        'دومينوز': 'دومينوز',
        'صب واي': 'صب واي',
        'صبواي': 'صب واي',
        'برجر كنج': 'برجر كنج',
        'برجركنج': 'برجر كنج'
    }

def clean_extracted_text(text: str) -> str:
    """
    Clean extracted restaurant text
    """
    if not text:
        return ""
    
    # Remove common prefixes and suffixes
    text = re.sub(r'^(مطعم|مقهى|كافيه|إلى|من|في|عند)\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*(من فضلك|لو سمحت|شكرا)$', '', text, flags=re.IGNORECASE)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove very short or common words
    excluded_words = ['نعم', 'لا', 'شكرا', 'أهلا', 'مرحبا', 'السلام', 'عليكم', 'وعليكم']
    if text.lower() in [word.lower() for word in excluded_words]:
        return ""
    
    return text

# Test function
def test_enhancements():
    """
    Test the enhancement functions
    """
    test_cases = [
        "مرحباً بك في محطة سابك",
        "محطة لاعربي",
        "أقرب مسار إلى مطعم البيك",
        "تفضل ما اسم المطعم",
        "الملك عبدالله إلى الرياض"
    ]
    
    print("🧪 Testing Arabic enhancements:")
    print("=" * 50)
    
    for test in test_cases:
        enhanced = enhance_arabic_pronunciation(test)
        clean = remove_diacritics(enhanced)
        print(f"Original: {test}")
        print(f"Enhanced: {enhanced}")
        print(f"Clean:    {clean}")
        print("-" * 30)

if __name__ == "__main__":
    test_enhancements()