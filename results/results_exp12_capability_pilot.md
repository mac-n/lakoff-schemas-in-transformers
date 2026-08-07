# exp12 — Pythia 70m capability sanity check

Schema sentences: 15, neutral controls: 15
Per layer, computing mean SAE-feature activation across token positions per sentence,
then per-feature mean differences between schema and neutral groups.

## Layer 0 (res-sm)
- schema-group mean activation across all features: 0.0005
- neutral-group mean activation: 0.0005
- fraction of features schema>neutral: 0.063

### Top features by |mean activation difference|
- feat 21053: diff=-0.1518 (tstat=-0.98) → neutral
  *the occurrences of the word "on."*
- feat 26535: diff=-0.0904 (tstat=-0.70) → neutral
  *the phrase "at" followed by a number or location*
- feat 11082: diff=+0.0849 (tstat=+0.54) → schema
  *references to investigations and legal proceedings involving significant issues*
- feat 28368: diff=+0.0814 (tstat=+0.38) → schema
  *the recurring mention of a specific year*
- feat 20110: diff=-0.0812 (tstat=-0.45) → neutral
  *themes related to familial relationships and responsibilities*
- feat 6073: diff=+0.0774 (tstat=+0.88) → schema
  *instances of the definite article "the"*
- feat 12967: diff=+0.0771 (tstat=+0.46) → schema
  *the past tense form of the verb "to be."*
- feat 25460: diff=-0.0769 (tstat=-0.55) → neutral
  *occurrences of the word "has" in various contexts*
- feat 28531: diff=+0.0746 (tstat=+1.16) → schema
  *phrases related to legal or procedural terminology*
- feat 17121: diff=-0.0727 (tstat=-0.79) → neutral
  *mentions of accommodations and their associated features*

### Top features by |t-statistic| (effect size adjusted)
- feat 23199: diff=-0.0403 (tstat=-1.26) → neutral
  *concepts related to personal growth, self-awareness, and community development*
- feat 13509: diff=-0.0306 (tstat=-1.22) → neutral
  *descriptions of processed food items and associated goods*
- feat 10952: diff=+0.0233 (tstat=+1.18) → schema
  *instances of personal pronouns and their usage in sentences*
- feat 28531: diff=+0.0746 (tstat=+1.16) → schema
  *phrases related to legal or procedural terminology*
- feat 4258: diff=-0.0258 (tstat=-1.15) → neutral
  *details related to product specifications and features*
- feat 2787: diff=-0.0322 (tstat=-1.13) → neutral
  *references to movies and music events*
- feat 6836: diff=-0.0115 (tstat=-1.13) → neutral
  *references to legal terms and procedural language*
- feat 31720: diff=+0.0040 (tstat=+1.10) → schema
  *negations and expressions of uncertainty*
- feat 26140: diff=-0.0096 (tstat=-1.09) → neutral
  *references to questions or inviting dialogue*
- feat 16010: diff=-0.0053 (tstat=-1.06) → neutral
  *references to making a positive impact or difference in various contexts*

## Layer 1 (res-sm)
- schema-group mean activation across all features: 0.0010
- neutral-group mean activation: 0.0012
- fraction of features schema>neutral: 0.076

### Top features by |mean activation difference|
- feat 26476: diff=-0.2124 (tstat=-0.86) → neutral
  *specific numerical values or statistics related to programming or technical details*
- feat 4525: diff=+0.1323 (tstat=+0.80) → schema
  *emotions and actions related to interpersonal interactions*
- feat 8816: diff=-0.1311 (tstat=-0.50) → neutral
  *references to specific days of the week*
- feat 23258: diff=-0.1191 (tstat=-0.68) → neutral
  *sentences that report information or provide details*
- feat 18460: diff=-0.0903 (tstat=-0.65) → neutral
  *references to specific days, especially "yesterday" and "today."*
- feat 18354: diff=+0.0823 (tstat=+0.76) → schema
  *references to regulations and guidelines related to education*
- feat 23346: diff=-0.0772 (tstat=-0.39) → neutral
  *terms related to anthropology and its subfields*
- feat 1238: diff=-0.0672 (tstat=-1.01) → neutral
  *instances of the word "on" in various contexts*
- feat 7158: diff=-0.0665 (tstat=-0.94) → neutral
  *occurrences of the word "on"*
- feat 20576: diff=-0.0641 (tstat=-0.55) → neutral
  *forms of the verb "have."*

### Top features by |t-statistic| (effect size adjusted)
- feat 9073: diff=-0.0104 (tstat=-1.43) → neutral
  *references to DJs and their performances*
- feat 26227: diff=-0.0478 (tstat=-1.38) → neutral
  *references to hospitality and leisure facilities*
- feat 11585: diff=-0.0195 (tstat=-1.37) → neutral
  *elements related to artistic techniques and formal analysis in visual art*
- feat 25123: diff=-0.0162 (tstat=-1.35) → neutral
  *numerical data and statistics*
- feat 12796: diff=-0.0149 (tstat=-1.34) → neutral
  *phrases indicating important entities or subjects in a text*
- feat 3020: diff=-0.0110 (tstat=-1.31) → neutral
  *time-related phrases and schedule indicators*
- feat 3121: diff=-0.0043 (tstat=-1.29) → neutral
  *terms related to quality or excellence*
- feat 15057: diff=-0.0564 (tstat=-1.29) → neutral
  *punctuation marks, specifically periods*
- feat 2612: diff=-0.0153 (tstat=-1.26) → neutral
  *instances of the word "of" in various contexts*
- feat 10585: diff=-0.0227 (tstat=-1.23) → neutral
  *references to dates and times*

## Layer 2 (res-sm)
- schema-group mean activation across all features: 0.0017
- neutral-group mean activation: 0.0019
- fraction of features schema>neutral: 0.063

### Top features by |mean activation difference|
- feat 10757: diff=-0.7371 (tstat=-0.66) → neutral
  *formal declarations of liability or legal terms*
- feat 4770: diff=-0.6242 (tstat=-0.66) → neutral
  *instances of legal terminology or phrases within a text*
- feat 17660: diff=-0.4627 (tstat=-0.70) → neutral
  *instances of high emotional or sentimental expression*
- feat 17961: diff=-0.2742 (tstat=-0.35) → neutral
  *proper nouns or significant names*
- feat 31010: diff=-0.2535 (tstat=-0.63) → neutral
  *programming constructs or elements relevant to software development*
- feat 26806: diff=-0.2362 (tstat=-0.72) → neutral
  *instances of specific formatting or structural elements in the text*
- feat 3717: diff=-0.2038 (tstat=-0.71) → neutral
  *instances of significant numbers or ratings*
- feat 23827: diff=-0.2010 (tstat=-0.37) → neutral
  *key action words and important subjects in various contexts*
- feat 13112: diff=-0.1766 (tstat=-0.43) → neutral
  *mentions of days of the week*
- feat 1302: diff=+0.1237 (tstat=+0.73) → schema
  *dialogue and interactions between characters*

### Top features by |t-statistic| (effect size adjusted)
- feat 31908: diff=+0.0252 (tstat=+1.47) → schema
  *words associated with actions and outcomes*
- feat 6469: diff=-0.0088 (tstat=-1.39) → neutral
  *historical references involving Austria and Hungary, particularly in the context of political alliances and events from World War I*
- feat 21047: diff=-0.0134 (tstat=-1.30) → neutral
  *terms and structures related to legal and administrative contexts, particularly involving articles, sections, and formal documents*
- feat 12158: diff=-0.0154 (tstat=-1.22) → neutral
  *detailed descriptions and specifications of products, particularly focusing on measurements and materials*
- feat 10741: diff=-0.0089 (tstat=-1.22) → neutral
  *phrases that discuss website usage and privacy-related terms*
- feat 12625: diff=+0.0201 (tstat=+1.17) → schema
  *elements related to atmosphere and experience*
- feat 17728: diff=+0.0276 (tstat=+1.16) → schema
  *phrases related to problems and their potential solutions*
- feat 21741: diff=-0.0085 (tstat=-1.16) → neutral
  *companies and their financial details or activities*
- feat 6285: diff=+0.0247 (tstat=+1.15) → schema
  *phrases related to institutional and organizational structures*
- feat 21218: diff=-0.0080 (tstat=-1.10) → neutral
  *instances of legal terminology and criminal charges*

## Layer 3 (res-sm)
- schema-group mean activation across all features: 0.0023
- neutral-group mean activation: 0.0025
- fraction of features schema>neutral: 0.049

### Top features by |mean activation difference|
- feat 5355: diff=-0.7199 (tstat=-0.70) → neutral
  *legal terminology related to liability and warranties*
- feat 21809: diff=-0.7195 (tstat=-0.72) → neutral
  *expressions of gratitude and friendship*
- feat 1812: diff=-0.5284 (tstat=-0.68) → neutral
  *complex legal terms and organizational identifiers*
- feat 3637: diff=-0.3790 (tstat=-0.70) → neutral
  *key terms related to legal proceedings and opinions*
- feat 20526: diff=-0.3246 (tstat=-0.45) → neutral
  *phrases that denote significant actions or noteworthy subjects*
- feat 22622: diff=-0.2291 (tstat=-0.75) → neutral
  *instances of line breaks or empty sections in the text*
- feat 26019: diff=-0.2104 (tstat=-0.48) → neutral
  *specific days of the week or notable dates*
- feat 12305: diff=+0.1662 (tstat=+0.96) → schema
  *intense emotional interactions or physical confrontations*
- feat 7254: diff=-0.1536 (tstat=-0.49) → neutral
  *key action words indicating significant events or actions*
- feat 32230: diff=-0.1301 (tstat=-0.67) → neutral
  *references to specific days of the week and dates*

### Top features by |t-statistic| (effect size adjusted)
- feat 6885: diff=-0.0329 (tstat=-2.10) → neutral
  *phrases related to time and events occurring or planned for the near future*
- feat 2734: diff=+0.0881 (tstat=+1.48) → schema
  *verbs and actions that depict movement or changes in state*
- feat 20621: diff=-0.0438 (tstat=-1.44) → neutral
  *legal terminology and phrases related to liability and warranties*
- feat 25976: diff=+0.0331 (tstat=+1.44) → schema
  *references to legal claims or court proceedings*
- feat 32687: diff=+0.0990 (tstat=+1.39) → schema
  *the definite article "the" across various contexts*
- feat 13804: diff=-0.0226 (tstat=-1.35) → neutral
  *terms related to contributions and effects in various contexts*
- feat 11677: diff=-0.0194 (tstat=-1.32) → neutral
  *mentions of financial and consumer advocacy*
- feat 32236: diff=-0.0515 (tstat=-1.31) → neutral
  *references to tags or archives in content*
- feat 20165: diff=-0.0328 (tstat=-1.29) → neutral
  *phrases or sentences that emphasize strong feelings or impactful statements*
- feat 12359: diff=-0.0334 (tstat=-1.28) → neutral
  *HTML document structure and related tags*

## Layer 4 (res-sm)
- schema-group mean activation across all features: 0.0014
- neutral-group mean activation: 0.0016
- fraction of features schema>neutral: 0.057

### Top features by |mean activation difference|
- feat 32255: diff=-0.5396 (tstat=-0.75) → neutral
  *references to legal proceedings or court cases*
- feat 9666: diff=-0.5393 (tstat=-0.75) → neutral
  *text related to legal or formal notices, particularly copyright and licensing information*
- feat 8286: diff=-0.3183 (tstat=-0.91) → neutral
  *instances of numerical or programming-related values such as parameters or settings*
- feat 28444: diff=-0.2944 (tstat=-0.81) → neutral
  *statements about disappointment or dissatisfaction*
- feat 5137: diff=-0.2285 (tstat=-0.54) → neutral
  *occurrences of specific dates or times*
- feat 27953: diff=-0.1514 (tstat=-0.61) → neutral
  *references to popular fried cheese dishes, particularly those associated with Wisconsin cuisine*
- feat 22775: diff=-0.1499 (tstat=-0.42) → neutral
  *proper nouns, names, and references to specific roles or positions*
- feat 12420: diff=+0.1369 (tstat=+0.26) → schema
  *conversations about mental health and self-perception*
- feat 18303: diff=-0.1358 (tstat=-0.89) → neutral
  *references to political and social issues*
- feat 27648: diff=+0.1344 (tstat=+1.11) → schema
  *references to Jewish communities and historical contexts*

### Top features by |t-statistic| (effect size adjusted)
- feat 25703: diff=-0.0203 (tstat=-1.42) → neutral
  *expressions of gratitude and appreciation*
- feat 27161: diff=-0.0540 (tstat=-1.32) → neutral
  *action verbs related to transactions or exchanges*
- feat 4925: diff=+0.1068 (tstat=+1.24) → schema
  *the presence of specific articles and nouns in a legal or formal context*
- feat 4365: diff=+0.0506 (tstat=+1.21) → schema
  *mentions of personal belongings and their sentimental value*
- feat 14425: diff=-0.0533 (tstat=-1.17) → neutral
  *references to articles or tags in document structures*
- feat 15964: diff=-0.0197 (tstat=-1.17) → neutral
  *questions about apps and their availability in app stores*
- feat 17478: diff=+0.0543 (tstat=+1.15) → schema
  *action verbs related to competition and effort in sports contexts*
- feat 3540: diff=-0.0384 (tstat=-1.13) → neutral
  *superlative adjectives or phrases indicating excellent qualities*
- feat 6225: diff=-0.0286 (tstat=-1.12) → neutral
  *phrases related to culinary experiences and cooking classes*
- feat 27648: diff=+0.1344 (tstat=+1.11) → schema
  *references to Jewish communities and historical contexts*

## Layer 5 (res-sm)
- schema-group mean activation across all features: 0.0010
- neutral-group mean activation: 0.0012
- fraction of features schema>neutral: 0.079

### Top features by |mean activation difference|
- feat 899: diff=-0.1900 (tstat=-0.19) → neutral
  *special characters and symbols*
- feat 27707: diff=-0.1858 (tstat=-0.68) → neutral
  *dates and times referenced in the text*
- feat 6591: diff=-0.1750 (tstat=-0.49) → neutral
  *occurrences of specific weekdays and dates*
- feat 3401: diff=-0.1562 (tstat=-0.54) → neutral
  *references to various types of restaurants and their culinary offerings*
- feat 30416: diff=-0.1553 (tstat=-0.48) → neutral
  *dates and days of the week*
- feat 11427: diff=+0.1368 (tstat=+0.59) → schema
  *statistical terms related to changes in numerical values, such as increases and decreases*
- feat 31333: diff=-0.1251 (tstat=-0.56) → neutral
  *keywords associated with theatrical and musical performances*
- feat 19695: diff=-0.1172 (tstat=-0.57) → neutral
  *references to specific conferences and professional gatherings*
- feat 31056: diff=-0.1103 (tstat=-0.97) → neutral
  *phrases indicating availability or options for booking and accessing services*
- feat 9990: diff=-0.1094 (tstat=-0.79) → neutral
  *phrases related to feelings of isolation or independence*

### Top features by |t-statistic| (effect size adjusted)
- feat 23141: diff=+0.0701 (tstat=+1.11) → schema
  *key moments of revelation or discovery about characters or situations*
- feat 2752: diff=-0.0458 (tstat=-1.07) → neutral
  *locations and personal characteristics of the individuals mentioned*
- feat 22547: diff=-0.0793 (tstat=-1.04) → neutral
  *elements of urban development and construction projects*
- feat 31253: diff=+0.0279 (tstat=+1.02) → schema
  *the word "the" in various contexts throughout the text*
- feat 353: diff=+0.0400 (tstat=+1.02) → schema
  *themes of conflict and struggle within human relationships*
- feat 24366: diff=-0.0466 (tstat=-0.98) → neutral
  *active verbs related to actions and processes in various contexts*
- feat 31056: diff=-0.1103 (tstat=-0.97) → neutral
  *phrases indicating availability or options for booking and accessing services*
- feat 14222: diff=+0.0418 (tstat=+0.97) → schema
  *themes related to spiritual struggle and the nature of good and evil*
- feat 24227: diff=-0.0718 (tstat=-0.96) → neutral
  *descriptions of buildings and their locations*
- feat 16819: diff=-0.0272 (tstat=-0.95) → neutral
  *locations and institutions relevant to cultural and educational events*

