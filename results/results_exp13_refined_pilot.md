# exp13 — Refined capability pilot: asymmetric grounding test

Sentences: 10 UP_LITERAL + 10 UP_MORE + 10 UP_HAPPY + 15 NEUTRAL.
Asymmetric grounding score = min(UP_LITERAL_mean, UP_MORE_mean, UP_HAPPY_mean) − NEUTRAL_mean.
Aggregation: MAX activation per feature across token positions per sentence.

Per layer: count of features with positive asym score, top 15 by score with descriptions.

## Summary across layers

  layer   n_pos_asym   n_strong_asym   top_asym_score
      0          210               0            0.488
      1          295               1            0.503
      2          333               1            0.600
      3          226               1            0.590
      4          246               1            0.637
      5          257               0            0.354

----------------------------------------------------------------------------------------------------
## Layer 0  (positive asym features: 210, strong: 0)
----------------------------------------------------------------------------------------------------

Top 15 features by asymmetric grounding score:
    feat    score   UP_LIT  UP_MORE  UP_HAPPY  NEUTRAL  description
    1635   +0.488    0.520    0.488     1.004    0.000  phrases indicating time references after significant events or situations
   28531   +0.289    1.360    0.610     0.623    0.321  phrases related to legal or procedural terminology
    6073   +0.262    1.473    0.817     1.098    0.555  instances of the definite article "the"
   14419   +0.126    0.450    0.599     0.897    0.324  xml elements and attributes within a structured document
   32149   +0.108    0.137    0.109     0.211    0.000  phrases reflecting repetition and time management
   21312   +0.091    0.258    0.130     0.129    0.038  sequences of chemical symbols and formulas
   10952   +0.087    0.320    0.241     0.314    0.154  instances of personal pronouns and their usage in sentences
    1986   +0.075    0.429    0.393     0.327    0.252  references to legal concepts and courtroom contexts
   27221   +0.075    0.246    0.467     0.311    0.172  discussions about gender dominance in various industries and their implications
   22802   +0.071    0.304    0.107     0.168    0.037  instances of legal terminology and related phrases
    2538   +0.065    0.402    0.345     0.395    0.280  attributes and characteristics related to target audiences and their needs
   24692   +0.065    0.108    0.081     0.396    0.015  instances where actions or events are initiated or completed
   23815   +0.064    0.374    0.143     0.160    0.079  certain programming keywords and errors related to code execution
   10235   +0.047    0.176    0.178     0.249    0.129  terms related to legal proceedings and criminal activities
    8342   +0.045    0.123    0.045     0.189    0.000  verbs related to actions of transfer, movement, or abrupt changes

----------------------------------------------------------------------------------------------------
## Layer 1  (positive asym features: 295, strong: 1)
----------------------------------------------------------------------------------------------------

Top 15 features by asymmetric grounding score:
    feat    score   UP_LIT  UP_MORE  UP_HAPPY  NEUTRAL  description
   18354   +0.503    1.664    1.246     1.478    0.743  references to regulations and guidelines related to education
   17164   +0.457    0.457    0.516     1.002    0.000  the phrase "after" in various contexts
    8436   +0.186    0.253    0.295     0.365    0.067  references to government actions and legal issues related to public service
    3151   +0.183    0.867    0.917     0.869    0.684  references to financial aspects and consequences in discussions about economic policies
   18996   +0.148    0.329    0.174     0.463    0.025  terms related to medical procedures and outcomes
   25962   +0.146    0.702    0.725     0.910    0.556  references to historical events or conditions
   16025   +0.128    0.485    0.279     0.257    0.129  specific technical components and hardware related to machinery and devices
   17796   +0.122    0.541    0.307     0.396    0.185  references to specific organizations, policies, or legal frameworks
   28917   +0.099    0.285    0.099     0.522    0.000  past-tense verbs and their variations in different forms
     363   +0.098    0.609    0.411     0.471    0.314  references to academic or educational contexts
    6981   +0.094    0.104    0.240     0.182    0.010  articles and determiners in various contexts
    9581   +0.093    0.554    0.118     0.221    0.026  past tense verbs indicating actions of people
   24247   +0.091    0.124    0.197     0.271    0.033  topics related to health, specifically illnesses and their impacts
    4165   +0.086    0.539    0.224     0.374    0.138  terms related to psychological conditions and their social implications
   28187   +0.084    0.368    0.503     0.084    0.000  specific terms related to parenting support and childbirth resources

----------------------------------------------------------------------------------------------------
## Layer 2  (positive asym features: 333, strong: 1)
----------------------------------------------------------------------------------------------------

Top 15 features by asymmetric grounding score:
    feat    score   UP_LIT  UP_MORE  UP_HAPPY  NEUTRAL  description
   14910   +0.600    1.572    1.176     1.339    0.576  references to the word "the" in various contexts
    4554   +0.437    0.437    0.473     0.973    0.000  repeated phrases indicating a time frame or chronological sequence
   29792   +0.189    0.206    0.293     0.271    0.017  phrases related to time duration and measurement
   31908   +0.156    0.363    0.273     0.168    0.012  words associated with actions and outcomes
   27894   +0.127    0.921    0.414     0.345    0.217  verbs related to significant actions or transformations
   15247   +0.113    0.191    0.180     0.184    0.066  elements related to completion and results in processes or tasks
   26381   +0.097    1.011    0.154     0.282    0.057  prepositions and their associated phrases in relation to spatial positions
   25321   +0.090    0.116    0.121     0.100    0.010  terms related to specific attributes or categories in various contexts
   15906   +0.079    0.091    0.142     0.121    0.012  references to legal or procedural issues related to rights and regulations
    2539   +0.078    0.096    0.117     0.211    0.018  legal and judicial language
   23500   +0.075    0.188    0.159     0.170    0.084  specific product features and attributes
   25643   +0.071    0.212    0.237     0.071    0.000  terms related to fundraising and the act of raising money
    2260   +0.070    0.136    0.079     0.126    0.009  items related to technical instructions or procedural steps
   16325   +0.068    0.102    0.083     0.079    0.011  HTML and XML tags and structure
   31733   +0.067    0.752    0.840     0.829    0.684  specific syntax or formatting elements in code

----------------------------------------------------------------------------------------------------
## Layer 3  (positive asym features: 226, strong: 1)
----------------------------------------------------------------------------------------------------

Top 15 features by asymmetric grounding score:
    feat    score   UP_LIT  UP_MORE  UP_HAPPY  NEUTRAL  description
   32687   +0.590    1.320    1.047     1.060    0.457  the definite article "the" across various contexts
    2734   +0.455    1.689    0.717     0.858    0.262  verbs and actions that depict movement or changes in state
    5883   +0.431    0.431    0.588     0.943    0.000  phrases indicating a delay or a period of time following an event
   23118   +0.265    6.916    7.688     7.561    6.651  proper nouns and specific identifiers in text
   22160   +0.251    2.791    3.380     3.227    2.540  instances of high numerical values or significant data points
    9761   +0.196    0.283    0.805     0.281    0.085  programming-related keywords and functions
    2654   +0.122    0.150    0.294     0.192    0.027  references to collaborative actions and shared experiences
    2398   +0.122    2.350    1.067     1.863    0.945  references to the letter "G."
   19729   +0.113    2.273    0.805     0.618    0.504  occurrences of the letter 'A' as a standalone character
   12319   +0.102    0.145    0.116     0.240    0.014  actions related to legal or legislative procedures
   15714   +0.100    0.331    0.224     0.100    0.000  phrases related to raising awareness or funds for causes
    2179   +0.099    0.153    0.165     0.099    0.000  verbs related to quitting or stopping actions
      93   +0.097   11.398   11.629    11.631   11.301  punctuation-like or symbolic elements in the text
    8489   +0.094    0.111    0.104     0.151    0.011  elements related to structure in written content, particularly coding or formatting tags
   17584   +0.092    0.588    0.824     0.929    0.496  content related to discussions or commentary on social issues and community engagement

----------------------------------------------------------------------------------------------------
## Layer 4  (positive asym features: 246, strong: 1)
----------------------------------------------------------------------------------------------------

Top 15 features by asymmetric grounding score:
    feat    score   UP_LIT  UP_MORE  UP_HAPPY  NEUTRAL  description
    4925   +0.637    1.205    1.120     1.298    0.483  the presence of specific articles and nouns in a legal or formal context
    3005   +0.394    0.487    0.394     0.980    0.000  instances of the word "after" in various contexts
   26547   +0.318    2.818    3.443     3.649    2.500  distinctive phrases or structures within text
    9546   +0.303    1.720    2.446     1.627    1.324  specific essay topics or related content in written works
   23542   +0.238    0.392    0.452     0.344    0.106  proper nouns, particularly names and significant terms
    3741   +0.180    1.611    0.387     0.216    0.036  the indefinite article "a" and the capital letter "A" signaling the beginning of significant statements or titles
   28368   +0.170    0.206    0.544     0.190    0.020  technical terms related to software development and programming
   19445   +0.163    0.289    0.231     0.279    0.068  technical terms related to product testing and consumer safety
   29812   +0.163    0.186    0.420     0.205    0.023  references to names and notable individuals or figures
   17478   +0.153    1.056    0.296     0.367    0.143  action verbs related to competition and effort in sports contexts
    1540   +0.150    0.150    0.304     0.210    0.000  events or actions related to significant societal changes or historical moments
   16527   +0.150    0.183    0.353     1.795    0.033  words related to obligations and legal terminology
   31393   +0.147    1.041    2.220     1.006    0.859  elements related to document structure or metadata
   21612   +0.143    0.881    1.169     1.032    0.738  statements related to potential, trust, and improvement in various contexts
   29184   +0.110    0.854    1.758     0.821    0.711  terms related to finance and investment

----------------------------------------------------------------------------------------------------
## Layer 5  (positive asym features: 257, strong: 0)
----------------------------------------------------------------------------------------------------

Top 15 features by asymmetric grounding score:
    feat    score   UP_LIT  UP_MORE  UP_HAPPY  NEUTRAL  description
   21316   +0.354    0.765    0.381     1.008    0.026  instances of the word "after."
   24256   +0.324    0.324    0.348     0.552    0.000  instances of phrases that indicate a sequence of events
   20076   +0.238    0.596    0.634     0.725    0.358  questions and statements related to accountability and responsibility
   12832   +0.207    0.631    0.697     0.758    0.425  references to technical specifications and features in software-related contexts
   12582   +0.189    1.626    1.445     1.841    1.256  specific details related to legal agreements and their implications
   25496   +0.143    0.360    0.423     0.389    0.217  phrases relating to legal testimony and courtroom proceedings
    8907   +0.128    0.779    0.821     0.783    0.650  mentions of community involvement and social service activities
   30532   +0.126    0.265    0.144     0.177    0.019  references to legal frameworks and their implications
    1857   +0.124    0.163    0.281     0.233    0.039  references to medical procedures and their associated outcomes
   31199   +0.120    0.171    1.023     0.209    0.051  scientific and mathematical terms related to interactions and relationships
   13792   +0.114    0.232    0.223     0.204    0.090  historical events and figures related to leadership and political changes
    5487   +0.106    0.237    0.106     0.211    0.000  words and phrases related to discussions of issues, problems, and questioning
   25173   +0.101    0.258    0.101     0.228    0.000  terms related to users, customers, and their interactions or experiences with various services
   30369   +0.096    0.506    0.096     0.190    0.000  phrases indicating a sequence of events or actions that should be taken prior to a certain time or condition
   14671   +0.092    0.166    0.179     0.236    0.075  references to mathematical or computational structures

