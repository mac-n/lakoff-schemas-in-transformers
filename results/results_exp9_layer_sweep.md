====================================================================================================
exp9: SAE decoder PCA + Neuronpedia auto-interp lookup, layer sweep
====================================================================================================

Pipeline: decoder PCA → top-10 features per top-10 PC → Neuronpedia lookup.
Hypothesis after exp8: res-sm L5 organises around Pile-domain register (legal, biomedical, math, programming),
NOT image schemas. exp9 tests how this pattern generalises across layers/substrates.

####################################################################################################
# mlp-sm L5  (pythia-70m-deduped-mlp-sm / blocks.5.hook_mlp_out)
####################################################################################################

n_features=32,768, d_model=512
Participation ratio: 163.0  |  top 10 PC variance ratios: ['0.0598', '0.0185', '0.0141', '0.0109', '0.0095', '0.0086', '0.0062', '0.0054', '0.0049', '0.0045']
Cumulative variance of top 10: 0.1424

----------------------------------------------------------------------------------------------------
PC 0  (variance ratio = 0.0598, cumulative = 0.0598)
----------------------------------------------------------------------------------------------------
feat  7481  loading +0.9849  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/7481]
  desc: information related to exports and trade statistics  | maxAct≈5.9, n_top_acts=59  | pos_logits: ',' '\n' '?"' '?âĢĿ' ',"'
feat 30021  loading -0.9820  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/30021]
  desc: characters or symbols in a document that could represent specific formatting or code structures  | maxAct≈18.4, n_top_acts=64  | pos_logits: '\nĉ     ' 'č\n                       ' '\n\n\n   ' '\n          ' '                  '
feat  6850  loading -0.9782  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/6850]
  desc: special characters and symbols in the text  | maxAct≈52.5, n_top_acts=68  | pos_logits: '\nĉ     ' '\n\n\n   ' 'č\n                       ' '\n  âĢĥ' '                  '
feat 26357  loading +0.9769  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/26357]
  desc: references to information, reminders, or articles  | maxAct≈16.7, n_top_acts=60  | pos_logits: '\n' '?"' '?' '  ' '")'
feat  3767  loading +0.9746  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/3767]
  desc: terms and topics related to financial measurements and assessments  | maxAct≈15.0, n_top_acts=62  | pos_logits: '\n' '?' '?"' "?'" ' _'
feat 29320  loading -0.9740  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/29320]
  desc: patterns related to non-English characters or symbols  | maxAct≈24.6, n_top_acts=67  | pos_logits: '\nĉ     ' 'č\n                       ' '\n\n\n   ' '\n          ' '\n  âĢĥ'
feat  1438  loading +0.9729  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/1438]
  desc: references to structured data or programming elements  | maxAct≈17.0, n_top_acts=60  | pos_logits: '\n' '?"' '?' '**' ' _'
feat 15388  loading +0.9678  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/15388]
  desc: problematic situations and the frustration that comes from them  | maxAct≈14.9, n_top_acts=63  | pos_logits: '?"' '"' " ':" '?' '**'
feat  3758  loading -0.9425  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/3758]
  desc: encoded characters or symbols typically found in programming or markup languages  | maxAct≈3.1, n_top_acts=67  | pos_logits: '\nĉ     ' '\n\n\n   ' 'č\n                       ' '<|outofrange|>' '\n  âĢĥ'
feat 28784  loading -0.9298  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/28784]
  desc: legal terminology and references to court proceedings  | maxAct≈2.0, n_top_acts=70  | pos_logits: 'č\n                       ' '<|outofrange|>' '\nĉ     ' '\n          ' '\n                                              '

----------------------------------------------------------------------------------------------------
PC 1  (variance ratio = 0.0185, cumulative = 0.0783)
----------------------------------------------------------------------------------------------------
feat 10267  loading -0.4400  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/10267]
  desc: email addresses, particularly those associated with Gmail  | maxAct≈2.3, n_top_acts=60  | pos_logits: '.' '.\\' '.*;' '.$$' '\\.'
feat 22965  loading -0.4325  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/22965]
  desc: multiple instances of semicolons and curly braces, indicating structured programming elements or configurations  | maxAct≈9.0, n_top_acts=64  | pos_logits: '\n\n\n\n' '\n\n' 'inary' '\n\n \n' '\nĉ\n'
feat 28711  loading -0.4318  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/28711]
  desc: references to legal citations and case numbers  | maxAct≈13.3, n_top_acts=30  | pos_logits: '.' '.]{}' 'icions' '.:' 'icolor'
feat 18562  loading -0.4285  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/18562]
  desc: sections that signal separation or categorization in content  | maxAct≈15.1, n_top_acts=30  | pos_logits: '\nĉ' '\nĉĉĉĉĉĉ' '\nĉĉĉĉ' '\n' '\nĉ\n'
feat 29384  loading -0.4254  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/29384]
  desc: references to specific dates or numerical data  | maxAct≈2.2, n_top_acts=55  | pos_logits: '-' 'âĢĲ' '-)' '--' '-,'
feat 27872  loading -0.4152  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/27872]
  desc: mathematical variables and numerical expressions  | maxAct≈10.4, n_top_acts=58  | pos_logits: 'asting' 'reed' 'orest' "''(" 'ben'
feat 18767  loading -0.4123  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/18767]
  desc: instances of the word "known" and its variations, indicating a focus on recognition or reputation  | maxAct≈4.5, n_top_acts=55  | pos_logits: ' also' ' smo' ' annually' ' hereafter' '-->'
feat 14042  loading -0.4112  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/14042]
  desc: patterns of repetition and structure in phrases  | maxAct≈1.7, n_top_acts=63  | pos_logits: ' Coul' ' Reading' ' LHC' ' Biosystems' ' Justice'
feat  5789  loading -0.4035  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/5789]
  desc: references to programming structures and conditions  | maxAct≈1.7, n_top_acts=55  | pos_logits: 'osing' 'het' 'hi' ' Tut' 'ro'
feat  4778  loading -0.4030  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/4778]
  desc: names and titles in legal contexts  | maxAct≈6.3, n_top_acts=60  | pos_logits: 'adata' 'own' 'ariable' 'orters' 'asan'

----------------------------------------------------------------------------------------------------
PC 2  (variance ratio = 0.0141, cumulative = 0.0924)
----------------------------------------------------------------------------------------------------
feat 29327  loading +0.8016  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/29327]
  desc: references to features and benefits of products  | maxAct≈1.9, n_top_acts=64  | pos_logits: ' coming' ' familiar' ' followed' ' promoted' ' Works'
feat   531  loading -0.7890  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/531]
  desc: elements related to structured or formatted text or code snippets  | maxAct≈2.9, n_top_acts=60  | pos_logits: '\nČ      ' '            \n ' '\n                                      ' '         ' '\n\n                                         '
feat 26387  loading +0.7889  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/26387]
  desc: references to data structures or programming concepts related to arrays and their properties  | maxAct≈3.5, n_top_acts=70  | pos_logits: '\n\n        ' '                                                             ' '\n\n      ' '\n                                              ' '                                     '
feat 27909  loading +0.7730  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/27909]
  desc: objects related to tools and their usage  | maxAct≈1.8, n_top_acts=59  | pos_logits: 'EXPORT' 'Illustration' 'Simplify' 'INSERT' 'protobuf'
feat  2396  loading +0.7602  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/2396]
  desc: programming language syntax and definitions  | maxAct≈2.0, n_top_acts=63  | pos_logits: 'tight' 'acin' 'SUM' 'RESULT' 'CUR'
feat 30930  loading +0.7468  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/30930]
  desc: references to programming or coding structures and functions  | maxAct≈1.7, n_top_acts=67  | pos_logits: '"_' '":' ')",' '",' '.",'
feat  7387  loading +0.7403  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/7387]
  desc: specific terms related to streaming and technical instructions  | maxAct≈2.3, n_top_acts=59  | pos_logits: 'apa' 'ariant' '^).' '^{' 'igraph'
feat 30515  loading +0.7172  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/30515]
  desc: references to specific numbered items or steps in a list  | maxAct≈3.1, n_top_acts=64  | pos_logits: 'oons' ' intervals' ')âĪĴ' ')' ')&'
feat 30550  loading +0.7142  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/30550]
  desc: numerical values and mathematical expressions related to computational processes  | maxAct≈4.7, n_top_acts=64  | pos_logits: 'fig' 'hep' 'rut' 'IQR' 'nes'
feat 23257  loading +0.7137  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/23257]
  desc: numerical values, particularly those related to scientific measurements  | maxAct≈1.8, n_top_acts=64  | pos_logits: 'BER' 'CAD' 'etting' 'ATES' 'UES'

----------------------------------------------------------------------------------------------------
PC 3  (variance ratio = 0.0109, cumulative = 0.1033)
----------------------------------------------------------------------------------------------------
feat 11615  loading +0.7783  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/11615]
  desc: phrases indicating descriptions or characteristics of subjects within a text  | maxAct≈2.7, n_top_acts=65  | pos_logits: 'ftware' 'âķĲâķĲâķĲâķĲ' ' 00000000000000000000000000000000' 'à´' '1451450014514500'
feat 12420  loading +0.7566  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/12420]
  desc: occurrences of the word "the" and the command "mkdir"  | maxAct≈11.3, n_top_acts=30  | pos_logits: '1451450014514500' 'à´' ' hand' ' tolerated' 'achus'
feat 31213  loading +0.7509  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/31213]
  desc: greetings or conversational openings  | maxAct≈29.1, n_top_acts=53  | pos_logits: 'ĥ½' '1451450014514500' 'Ī' '=====================' 'ī'
feat 30411  loading +0.7070  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/30411]
  desc: phrases related to opportunities and experiences  | maxAct≈3.8, n_top_acts=40  | pos_logits: '1451450014514500' '=====================' 'Ī' 'ĭ' 'º'
feat  1541  loading +0.7000  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/1541]
  desc: specific instances of common introductory words and phrases  | maxAct≈6.9, n_top_acts=30  | pos_logits: ' tolerated' ' strat' ' grate' '1451450014514500' ' regeneration'
feat 15712  loading +0.6846  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/15712]
  desc: references to essay topics and their significance  | maxAct≈6.3, n_top_acts=66  | pos_logits: '=====================' '=======================' '=================================' 'ADVERTISEMENT' '------------------------'
feat  5525  loading +0.6763  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/5525]
  desc: technical terms and specific variable names related to programming and data structures  | maxAct≈5.0, n_top_acts=52  | pos_logits: ':")' ' 00000000000000000000000000000000' 'Intent' '_)' '()).'
feat 23945  loading +0.6703  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/23945]
  desc: elements related to legal or judicial documentation  | maxAct≈4.3, n_top_acts=66  | pos_logits: '=====================' '===============' 'href' 'maker' '---------------------'
feat 24657  loading +0.6699  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/24657]
  desc: specific identifiers and parameters in programming contexts  | maxAct≈1.3, n_top_acts=44  | pos_logits: '\n\n         ' '                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                ' 'č\n                   ' '<|outofrange|>' '                       '
feat 22796  loading +0.6618  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/22796]
  desc: phrases related to deception and manipulation  | maxAct≈2.2, n_top_acts=49  | pos_logits: 'ĥ½' 'Ī' '¹' 'Ł' '®'

----------------------------------------------------------------------------------------------------
PC 4  (variance ratio = 0.0095, cumulative = 0.1128)
----------------------------------------------------------------------------------------------------
feat 10151  loading -0.5984  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/10151]
  desc: variables or identifiers related to scientific data or coding contexts  | maxAct≈2.2, n_top_acts=64  | pos_logits: 'jcmm' 'untime' 'ALSE' 'Ãºblic' 'OOGLE'
feat 30800  loading -0.5698  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/30800]
  desc: requests for information or action  | maxAct≈2.3, n_top_acts=60  | pos_logits: 'please' 'thank' '======' 'trade' 'pay'
feat 14126  loading -0.5654  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/14126]
  desc: technical command outputs related to software or system operations  | maxAct≈2.7, n_top_acts=61  | pos_logits: 'AndroidRuntime' 'msgstr' '--------------------------------------------------------------------------------------------------------------------------------' '----------------------------------------------------------------------------------------------------------------' 'Calculate'
feat 25552  loading -0.5453  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/25552]
  desc: inquiries related to data processing and programming concepts  | maxAct≈11.9, n_top_acts=47  | pos_logits: 'ł' '¹' '²' '³' '°'
feat  5093  loading -0.5444  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/5093]
  desc: numeric values or parameters in a dataset or code context  | maxAct≈4.4, n_top_acts=57  | pos_logits: 'iNdEx' 'supplementary' 'medsc' 'sid' 'Č'
feat 24076  loading -0.5437  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/24076]
  desc: C/C++ preprocessor directives and header file inclusions  | maxAct≈7.0, n_top_acts=65  | pos_logits: 'dAtA' 'medsc' 'doibase' 'ps' 'gpu'
feat 19882  loading -0.5368  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/19882]
  desc: programming-related queries and instructions  | maxAct≈7.0, n_top_acts=60  | pos_logits: 'ł' 'Ĩ' '²' '¸' '¹'
feat 32647  loading -0.5342  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/32647]
  desc: code snippets related to user interface elements and interactions  | maxAct≈2.8, n_top_acts=63  | pos_logits: 'Ùģ' 'fraction' 'medsc' 'oddsidemargin' 'ÙĪØ¯'
feat  1892  loading -0.5285  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/1892]
  desc: technical commands and script syntax related to programming or scripting languages  | maxAct≈4.8, n_top_acts=59  | pos_logits: 'cd' 'cp' 'apt' 'cat' 'mv'
feat  8653  loading -0.5266  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/8653]
  desc: references to legal issues or justice-related matters  | maxAct≈9.5, n_top_acts=69  | pos_logits: 'ADVERTISEMENT' 'His' 'Both' 'READ' 'He'

----------------------------------------------------------------------------------------------------
PC 5  (variance ratio = 0.0086, cumulative = 0.1214)
----------------------------------------------------------------------------------------------------
feat  4342  loading +0.7450  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/4342]
  desc: structured data and code elements related to identity and information systems  | maxAct≈1.9, n_top_acts=64  | pos_logits: '·¸' 'ĻĤ' '\n\n               ' '<|outofrange|>' '                                         '
feat  6273  loading +0.6967  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/6273]
  desc: code structure and function definitions in programming contexts  | maxAct≈2.6, n_top_acts=65  | pos_logits: '#{$' '000001' 'charset' 'Sprintf' 'namespace'
feat 25177  loading +0.6691  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/25177]
  desc: structured numerical patterns and sequences in the text  | maxAct≈1.2, n_top_acts=58  | pos_logits: 'Made' 'Docket' 'Asked' 'xhtml' 'Sprintf'
feat  7567  loading +0.6292  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/7567]
  desc: JSON-like structured data elements and their properties  | maxAct≈6.5, n_top_acts=62  | pos_logits: '³' '¢' 'ĺ' 'Ļ' '½'
feat 20850  loading +0.6209  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/20850]
  desc: references to specific names and quantities  | maxAct≈2.2, n_top_acts=63  | pos_logits: 'xymatrix' '+}' 'linewidth' 'xhtml' 'blockList'
feat 20376  loading +0.6171  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/20376]
  desc: references to a diverse range of options and amenities  | maxAct≈2.4, n_top_acts=69  | pos_logits: ' crime' ' literature' ' paraffin' ' caliber' 'hips'
feat 26513  loading +0.6050  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/26513]
  desc: programming-related terms and expressions  | maxAct≈2.1, n_top_acts=62  | pos_logits: 'ETHERTYPE' 'iNdEx' 'medsc' 'rbrack' 'end'
feat  7206  loading +0.5925  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/7206]
  desc: elements related to data processing and framework components  | maxAct≈6.2, n_top_acts=66  | pos_logits: 'svg' 'interface' '#{$' 'linewidth' 'iNdEx'
feat 21796  loading -0.5881  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/21796]
  desc: phrases that indicate agreement, satisfaction, or relationships in various contexts  | maxAct≈1.0, n_top_acts=59  | pos_logits: ' gratefully' ' operation' '[]{' ' pleased' ' trouble'
feat 29016  loading +0.5769  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/29016]
  desc: elements related to code structure and dependencies in software packages  | maxAct≈2.7, n_top_acts=67  | pos_logits: '         ' '<|outofrange|>' 'č\n         ' '<|outofrange|>' '        \n '

----------------------------------------------------------------------------------------------------
PC 6  (variance ratio = 0.0062, cumulative = 0.1276)
----------------------------------------------------------------------------------------------------
feat 31317  loading -0.5907  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/31317]
  desc: terms related to environmental concerns and regulations, particularly in the context of pollution and sustainability efforts  | maxAct≈2.2, n_top_acts=62  | pos_logits: ' there' ' decided' ' this' ' Superior' ' THERE'
feat 13733  loading -0.5838  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/13733]
  desc: themes related to betrayal and emotional hurt in relationships  | maxAct≈1.6, n_top_acts=61  | pos_logits: ' alike' '"?"' ' thereto' 'âĢ²,' 'ichever'
feat 13159  loading -0.5385  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/13159]
  desc: references to public health policies and related discussions  | maxAct≈1.4, n_top_acts=64  | pos_logits: '  \n ' '<|outofrange|>' '                                 ' '\n\n                   ' '<|outofrange|>'
feat 15560  loading -0.5365  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/15560]
  desc: legal terms and phrases related to court proceedings and judgments  | maxAct≈1.4, n_top_acts=57  | pos_logits: ']{}' '].$$' '\n\nĉ' '\nĉ\n' '\n\n'
feat 13075  loading -0.4988  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/13075]
  desc: phrases related to complaints or expressions of frustration  | maxAct≈1.8, n_top_acts=63  | pos_logits: ' manages' ' isn' ' works' ' blows' ' sounds'
feat 16523  loading -0.4971  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/16523]
  desc: programming-related entities and operations within code documentation  | maxAct≈2.0, n_top_acts=61  | pos_logits: ':`' ' =>' ' manually' ' AUTHOR' ' flag'
feat 15105  loading -0.4942  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/15105]
  desc: references to stock trading and financial analysis data  | maxAct≈1.7, n_top_acts=50  | pos_logits: 'ģ' '           \n ' 'č\n           ' '\nĉ   ' '    \n '
feat 31433  loading -0.4892  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/31433]
  desc: references to events or locations  | maxAct≈3.6, n_top_acts=65  | pos_logits: "'?" ' Against' ' Fiction' ' Away' ' Novel'
feat 31490  loading -0.4811  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/31490]
  desc: words and phrases related to sports events and ticket promotions  | maxAct≈1.2, n_top_acts=61  | pos_logits: '))/(-' 'msgstr' '\\)' 'Za' 'Figure'
feat 16450  loading -0.4800  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/16450]
  desc: specific years or dates mentioned in the text  | maxAct≈5.5, n_top_acts=65  | pos_logits: ' meets' '-' ' entitled' ' versus' ' as'

----------------------------------------------------------------------------------------------------
PC 7  (variance ratio = 0.0054, cumulative = 0.1330)
----------------------------------------------------------------------------------------------------
feat 32254  loading +0.4765  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/32254]
  desc: terms associated with devices, particularly in a technical context  | maxAct≈1.1, n_top_acts=56  | pos_logits: ']\\]).' '$)' '")' '"}).' ')."'
feat 26021  loading -0.4181  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/26021]
  desc: programming constructs and operations involving variables and return statements  | maxAct≈5.7, n_top_acts=67  | pos_logits: ' Transl' ' $("#' ' Variables' ' settings' ' Objects'
feat  6672  loading -0.4112  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/6672]
  desc: technical terms related to systems and their operation  | maxAct≈1.6, n_top_acts=55  | pos_logits: 'ĻĤ' '°' '³' '¯' 'ľĵ'
feat 31164  loading +0.3928  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/31164]
  desc: connections between actions and their consequences in various contexts  | maxAct≈0.8, n_top_acts=62  | pos_logits: '.]' '.)' ' their' ' thereof' ' accordingly'
feat 29823  loading -0.3869  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/29823]
  desc: terms related to legal and ethical considerations, especially in the context of content and data management  | maxAct≈0.9, n_top_acts=59  | pos_logits: '<|outofrange|>' '\n                                  ' '\n                                           ' '                                           ' '<|outofrange|>'
feat 17304  loading -0.3799  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/17304]
  desc: keywords and modifiers used in programming and class definitions  | maxAct≈8.5, n_top_acts=64  | pos_logits: '\n                 ' 'č\n    ' '<|outofrange|>' 'č\n      ' '\n           '
feat 28790  loading -0.3770  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/28790]
  desc: assignments and variable declarations in programming code  | maxAct≈5.3, n_top_acts=63  | pos_logits: ' Academic' ' ut' ' Character' ' Variable' ' Input'
feat  2632  loading -0.3765  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/2632]
  desc: questions and items related to sports statistics  | maxAct≈3.1, n_top_acts=65  | pos_logits: ' Fourteenth' ' Superior' ' Eleventh' ' Miranda' ' Quarter'
feat 14333  loading +0.3726  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/14333]
  desc: groups of individuals or entities that are engaged in activities or services  | maxAct≈3.8, n_top_acts=59  | pos_logits: ' around' ' happening' '/.' ' surrounded' ' involved'
feat 11691  loading +0.3658  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/11691]
  desc: references to religious groups and related discussions about beliefs and practices  | maxAct≈1.5, n_top_acts=63  | pos_logits: ' Are' ' Have' ' threaten' ' hesitate' ' thrive'

----------------------------------------------------------------------------------------------------
PC 8  (variance ratio = 0.0049, cumulative = 0.1380)
----------------------------------------------------------------------------------------------------
feat  5155  loading +0.4640  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/5155]
  desc: specific numerical data and statistical information related to health or real estate  | maxAct≈1.4, n_top_acts=64  | pos_logits: '765' '667' '565' '485' '577'
feat 24992  loading +0.4630  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/24992]
  desc: elements related to programming and mouse events  | maxAct≈1.1, n_top_acts=62  | pos_logits: ')' ' DOI' ').' ')](' ',)'
feat 12153  loading +0.4456  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/12153]
  desc: themes related to emotional impact and personal connection in texts  | maxAct≈1.4, n_top_acts=57  | pos_logits: ' Access' ' War' ' Library' ' Cour' ' F'
feat 14462  loading -0.4449  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/14462]
  desc: keywords and phrases associated with programming and technical documentation  | maxAct≈1.2, n_top_acts=52  | pos_logits: 'ģ' '§' 'ĳ' 'į' 'Ŀ'
feat 20025  loading -0.4378  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/20025]
  desc: concepts related to personal growth and self-improvement challenges  | maxAct≈1.4, n_top_acts=59  | pos_logits: ' unusual' ' pleasure' ' cases' ' deadlines' ' supervision'
feat  8817  loading -0.4272  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/8817]
  desc: references to emotional pain and human connections  | maxAct≈1.3, n_top_acts=57  | pos_logits: 'upon' 'for' 'of' 'such' 'to'
feat 20841  loading +0.4197  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/20841]
  desc: medical terminology related to treatments and surgical procedures involving platelets and cardiovascular conditions  | maxAct≈3.5, n_top_acts=57  | pos_logits: 'ĨĴ' '©' '\n          ' '               ' '                       '
feat  7071  loading -0.4174  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/7071]
  desc: words related to programming and technical jargon  | maxAct≈1.1, n_top_acts=55  | pos_logits: '·¸' '¿½' 'Ģ' 'ÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤ' '              '
feat 22133  loading +0.4061  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/22133]
  desc: mentions of influential political figures and events  | maxAct≈1.2, n_top_acts=58  | pos_logits: '\n                                         ' '                                                                     ' '                                                               ' '<|outofrange|>' '\n                                              '
feat 16835  loading +0.3992  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/16835]
  desc: references to formal actions or procedures  | maxAct≈7.8, n_top_acts=61  | pos_logits: ' Por' ' Pero' ' Nos' ' Sin' ' Os'

----------------------------------------------------------------------------------------------------
PC 9  (variance ratio = 0.0045, cumulative = 0.1424)
----------------------------------------------------------------------------------------------------
feat 11423  loading -0.4419  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/11423]
  desc: words related to the classification and characteristics of various subjects or entities  | maxAct≈1.6, n_top_acts=60  | pos_logits: '\n                               ' '                                       ' '                                        ' '                                                ' '                                                                             '
feat  1287  loading -0.4249  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/1287]
  desc: names related to historical or political figures and their actions  | maxAct≈1.8, n_top_acts=60  | pos_logits: ' \n       ' '\n                                       ' '                                    ' '                                                                          ' '                             '
feat 24811  loading -0.4249  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/24811]
  desc: proper nouns and names associated with individuals and events  | maxAct≈2.4, n_top_acts=60  | pos_logits: 'zy' 'emi' 'zo' 'unes' 'oses'
feat 25926  loading -0.3960  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/25926]
  desc: institutions, processes, and terminology related to law and governance  | maxAct≈0.8, n_top_acts=65  | pos_logits: '"?' '"' ')",' ')"' '?"'
feat 14268  loading -0.3775  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/14268]
  desc: names of people and references to geographic locations or governments  | maxAct≈1.0, n_top_acts=60  | pos_logits: '            ' '                 ' '\n  ' '\n              ' 'č\n                   '
feat  3638  loading +0.3750  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/3638]
  desc: topics related to job training and opportunities in various industries  | maxAct≈1.3, n_top_acts=56  | pos_logits: '¶' 'ī' 'ĻĤ' '·¸' '»'
feat  4120  loading +0.3694  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/4120]
  desc: programming-related terms and structures  | maxAct≈2.8, n_top_acts=54  | pos_logits: '>()' '})}' '>();' '_);' '());'
feat 29933  loading +0.3622  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/29933]
  desc: elements related to computer programming and code structure  | maxAct≈1.3, n_top_acts=56  | pos_logits: '»¿' '¿½' 'ł' '\nĉ     ' '<|outofrange|>'
feat 17426  loading +0.3598  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/17426]
  desc: operations related to creating and managing storage and database elements  | maxAct≈3.7, n_top_acts=61  | pos_logits: 'NFTA' '»¿' 'Ļ' '®' '¹'
feat 24526  loading +0.3594  [https://www.neuronpedia.org/pythia-70m-deduped/5-mlp-sm/24526]
  desc: terms related to feedback and responses in a communicative context  | maxAct≈0.9, n_top_acts=63  | pos_logits: '}$\\' '}?' '\\]](' '}}</' '.*;'

####################################################################################################
# res-sm L3  (pythia-70m-deduped-res-sm / blocks.3.hook_resid_post)
####################################################################################################

n_features=32,768, d_model=512
Participation ratio: 422.7  |  top 10 PC variance ratios: ['0.0144', '0.0098', '0.0093', '0.0061', '0.0056', '0.0050', '0.0045', '0.0043', '0.0041', '0.0039']
Cumulative variance of top 10: 0.0670

----------------------------------------------------------------------------------------------------
PC 0  (variance ratio = 0.0144, cumulative = 0.0144)
----------------------------------------------------------------------------------------------------
feat 20209  loading +0.8349  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/20209]
  desc: references to programming and technical specifications  | maxAct≈3.8, n_top_acts=70  | pos_logits: 'ictions' 'annels' ' held' ' alone' ' const'
feat 29565  loading +0.7827  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/29565]
  desc: references to emotional states and interpersonal relationships  | maxAct≈14.7, n_top_acts=67  | pos_logits: ' underlying' ' este' ' positively' ' fact' ' entire'
feat 12340  loading +0.7824  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/12340]
  desc: terms related to streaming services and media consumption  | maxAct≈5.1, n_top_acts=49  | pos_logits: ' Foundation' 'edia' 'inburgh' 'caster' 'ictionary'
feat  2831  loading +0.7802  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/2831]
  desc: references to legal principles or guidelines  | maxAct≈6.6, n_top_acts=68  | pos_logits: 'lights' ' positively' ' entire' 'rients' ' depending'
feat  2959  loading +0.7537  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/2959]
  desc: terms related to legal or contractual disputes  | maxAct≈6.2, n_top_acts=59  | pos_logits: 'ictions' 'inburgh' 'rive' 'ertain' 'anges'
feat 20526  loading +0.7516  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/20526]
  desc: phrases that denote significant actions or noteworthy subjects  | maxAct≈56.7, n_top_acts=39  | pos_logits: 'ancellor' ' spirits' 'rosis' 'acha' 'ensive'
feat 28269  loading -0.7475  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/28269]
  desc: phrases related to obtaining licenses or permissions  | maxAct≈12.9, n_top_acts=56  | pos_logits: ' permission' 'uego' ' approval' ' waived' ' privileges'
feat  7254  loading +0.7429  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/7254]
  desc: key action words indicating significant events or actions  | maxAct≈26.3, n_top_acts=37  | pos_logits: 'acha' 'ancellor' 'rosis' 'ensive' ' spirits'
feat  6353  loading -0.7304  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/6353]
  desc: punctuation and conjunctions in the text  | maxAct≈15.0, n_top_acts=47  | pos_logits: ' waived' 'undo' ' sob' 'except' 'unders'
feat 26771  loading -0.7267  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/26771]
  desc: terms related to legal compliance and licenses  | maxAct≈13.8, n_top_acts=40  | pos_logits: 'ģ' '·' '©' 'ŀ' 'Ī'

----------------------------------------------------------------------------------------------------
PC 1  (variance ratio = 0.0098, cumulative = 0.0242)
----------------------------------------------------------------------------------------------------
feat 12170  loading -0.6890  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/12170]
  desc: technical language related to computer programming and specifications  | maxAct≈2.5, n_top_acts=64  | pos_logits: ' fly' 'letter' 'agus' 'quet' 'itol'
feat 16801  loading +0.6655  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/16801]
  desc: references to educational institutions and their interactions within the community  | maxAct≈2.1, n_top_acts=54  | pos_logits: 'krit' 'à°¿' 'à¯ģ' 'iento' 'à®¿'
feat 12521  loading -0.6355  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/12521]
  desc: punctuation or sentence-ending markers  | maxAct≈5.4, n_top_acts=68  | pos_logits: ' suppose' 'coli' 'ammad' ' pardon' ' Redistributions'
feat 29785  loading -0.6254  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/29785]
  desc: verbs indicating actions or movements related to a subject or object  | maxAct≈5.2, n_top_acts=59  | pos_logits: ' safely' ' reliably' ' efficiently' ' stably' ' permanently'
feat 19846  loading +0.6211  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/19846]
  desc: phrases expressing emotions and curiosities about life experiences and relationships  | maxAct≈1.0, n_top_acts=59  | pos_logits: 'enny' '[/' '"));' 'yler' 'rax'
feat 16843  loading -0.6146  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/16843]
  desc: the word "as" in various contexts  | maxAct≈6.9, n_top_acts=67  | pos_logits: ' those' '\n                                           ' '\n\n\n   ' '                       ' '                                                                     '
feat 27168  loading -0.6091  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/27168]
  desc: occurrences of the word "to."  | maxAct≈7.2, n_top_acts=65  | pos_logits: ' distribute' ' reproduce' ' illustrate' ' describe' ' efficiently'
feat  7608  loading -0.6064  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/7608]
  desc: articles that suggest specificity or give a high degree of certainty  | maxAct≈3.3, n_top_acts=65  | pos_logits: ' dozen' ' priori' 'Ĵ' 'ĸ' ' plurality'
feat 18516  loading -0.5863  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/18516]
  desc: phrases indicating uncertainty or hesitation in communication  | maxAct≈1.6, n_top_acts=62  | pos_logits: '            ' '\n                                              ' '                                                                                                                                                                                                                                                                ' '                                      ' '\n\n\n   '
feat 14732  loading -0.5816  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/14732]
  desc: content related to online shopping and product availability  | maxAct≈1.3, n_top_acts=60  | pos_logits: ' yours' ' your' ' adorable' ' ðŁĻĤ' ' favorite'

----------------------------------------------------------------------------------------------------
PC 2  (variance ratio = 0.0093, cumulative = 0.0335)
----------------------------------------------------------------------------------------------------
feat  6516  loading +0.8267  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/6516]
  desc: programming language syntax and structure  | maxAct≈4.4, n_top_acts=62  | pos_logits: '»¿' '³' 'Ļª' 'ĻĤ' '¾'
feat  9922  loading -0.7840  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/9922]
  desc: numerical values and their variations, likely focusing on mathematical or statistical data  | maxAct≈5.2, n_top_acts=60  | pos_logits: 'upal' '=âĢĿ' 'lite' 'hip' 'writers'
feat 27082  loading -0.7822  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/27082]
  desc: code-related elements, especially those referencing function calls and parameters  | maxAct≈3.6, n_top_acts=60  | pos_logits: ' pill' ' writing' 'worthy' ' scheduled' ' liner'
feat 29135  loading -0.7625  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/29135]
  desc: phrases indicating conditions or requirements for something to occur  | maxAct≈2.9, n_top_acts=47  | pos_logits: '$/' 'azine' 'mill' '){#' 'wright'
feat 21853  loading -0.7523  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/21853]
  desc: instances of numeric values and related data attributes  | maxAct≈8.0, n_top_acts=60  | pos_logits: '$/' 'ressor' ':`' 'lessly' 'akespe'
feat 14362  loading -0.7471  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/14362]
  desc: phrases that involve comparisons and alternatives  | maxAct≈2.6, n_top_acts=58  | pos_logits: 'letter' 'opter' ' these' '?).' 'hip'
feat 26405  loading -0.6875  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/26405]
  desc: references to specific historical figures and titles  | maxAct≈3.9, n_top_acts=63  | pos_logits: 'akespe' 'ressor' 'Critical' '---|---|---' 'ĥ½'
feat 26526  loading -0.6868  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/26526]
  desc: code snippets and structure definitions  | maxAct≈3.1, n_top_acts=62  | pos_logits: ' pill' 'Simplify' 'how' ' accession' 'iver'
feat 28098  loading +0.6811  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/28098]
  desc: sequences of digits and numbers  | maxAct≈3.9, n_top_acts=64  | pos_logits: '·¸' '»¿' '³' '´' '¾'
feat 27740  loading +0.6470  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/27740]
  desc: phrases associated with food and cooking  | maxAct≈1.3, n_top_acts=65  | pos_logits: ' itself' 'aches' ' belongs' 'ists' ' respects'

----------------------------------------------------------------------------------------------------
PC 3  (variance ratio = 0.0061, cumulative = 0.0397)
----------------------------------------------------------------------------------------------------
feat 24696  loading +0.7346  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/24696]
  desc: syntactic structures involving programming commands and conditionals  | maxAct≈5.8, n_top_acts=65  | pos_logits: '_."' '\'"' 'nai' "']." 'ifice'
feat   999  loading -0.6841  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/999]
  desc: numbers associated with currency or financial transactions  | maxAct≈3.6, n_top_acts=62  | pos_logits: 'othal' 'umen' ' ashes' 'hog' 'lop'
feat 28479  loading +0.6739  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/28479]
  desc: markup or structure in formatted text  | maxAct≈7.1, n_top_acts=52  | pos_logits: "'):" 'ubicin' '"){' 'osing' "':"
feat  7813  loading -0.6342  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/7813]
  desc: mentions of sports teams  | maxAct≈8.9, n_top_acts=66  | pos_logits: 'lement' 'idden' 'itage' 'corn' 'aland'
feat 31683  loading +0.5819  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/31683]
  desc: terms related to health conditions and treatments  | maxAct≈3.4, n_top_acts=65  | pos_logits: ' etc' 'hips' 'ãģªãģ©' 'áĢ¬' 'çŃī'
feat  7323  loading +0.5769  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/7323]
  desc: technical references related to coding or programming errors  | maxAct≈2.7, n_top_acts=60  | pos_logits: 'watson' 'NOW' 'terminus' 'FUNCTION' 'OULD'
feat 21883  loading +0.5512  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/21883]
  desc: syntax and structure commonly used in programming or coding contexts  | maxAct≈1.7, n_top_acts=55  | pos_logits: 'due' 'leftrightarrow' ' forgiven' '\'"' 'ongs'
feat 23289  loading +0.5428  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/23289]
  desc: references to financial and business performance metrics  | maxAct≈3.6, n_top_acts=66  | pos_logits: ',' ',...' ',,' ',}' 'áĥĶ'
feat  4309  loading -0.5278  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/4309]
  desc: sequences of numerical values and their patterns  | maxAct≈5.7, n_top_acts=65  | pos_logits: '·¸' 'illac' 'lement' 'aland' '³'
feat 13363  loading -0.5242  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/13363]
  desc: configurations or structural elements in programming or data organization  | maxAct≈2.6, n_top_acts=64  | pos_logits: '»' '·¸' 'Ĵ' 'Ľ' 'Ŀ'

----------------------------------------------------------------------------------------------------
PC 4  (variance ratio = 0.0056, cumulative = 0.0452)
----------------------------------------------------------------------------------------------------
feat 25770  loading -0.7258  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/25770]
  desc: technical or programming-related terms and structures  | maxAct≈5.0, n_top_acts=60  | pos_logits: 'velopment' 'TRODUCTION' 'istration' 'isons' ' ashes'
feat 32727  loading +0.6338  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/32727]
  desc: references to structured data formats and programming constructs  | maxAct≈3.5, n_top_acts=65  | pos_logits: '¦' 'ĥ½' 'Ĥ¬' '<|outofrange|>' '               '
feat 10903  loading +0.6331  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/10903]
  desc: proper nouns, particularly names  | maxAct≈2.9, n_top_acts=59  | pos_logits: '¦' 'ı' 'ĻĤ' '¯' 'Ĵ'
feat  2752  loading +0.5923  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/2752]
  desc: phrases and terms related to legislative and regulatory processes  | maxAct≈2.6, n_top_acts=60  | pos_logits: 'Ĳ' 'FFIRMED' 'ĸ' '        \n ' '                                                                                                                                                                                                                                '
feat 21404  loading +0.5895  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/21404]
  desc: specific product descriptions and features  | maxAct≈1.8, n_top_acts=64  | pos_logits: 'Ĳ' ' society' '¿½' ' offence' ' inversion'
feat 12789  loading +0.5795  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/12789]
  desc: numerical values and parameters related to data structures or configurations  | maxAct≈3.3, n_top_acts=65  | pos_logits: '¦' '¢' 'Ĥ¬' '»¿' 'ĻĤ'
feat 31083  loading -0.5554  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/31083]
  desc: key concepts and terminology related to visual arts and photography  | maxAct≈6.3, n_top_acts=64  | pos_logits: 'lessly' 'eting' 'irie' 'mitter' '__.'
feat 29489  loading -0.5333  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/29489]
  desc: computer programming keywords and syntax elements  | maxAct≈2.3, n_top_acts=57  | pos_logits: 'htra' ' shotgun' ' quo' ')(' ' whistle'
feat 22007  loading +0.5284  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/22007]
  desc: topics related to copyright and intellectual property rights issues  | maxAct≈1.3, n_top_acts=61  | pos_logits: ' sure' ' posting' ' people' 'abouts' ' anything'
feat 21225  loading +0.5270  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/21225]
  desc: HTML and CSS code snippets related to forms  | maxAct≈2.3, n_top_acts=63  | pos_logits: 'Ã¢' 'ŀ' 'endl' 'Ľ' 'ī'

----------------------------------------------------------------------------------------------------
PC 5  (variance ratio = 0.0050, cumulative = 0.0502)
----------------------------------------------------------------------------------------------------
feat 26444  loading +0.4752  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/26444]
  desc: code snippets related to programming functions and parameters  | maxAct≈2.7, n_top_acts=64  | pos_logits: '©' '£' "']." '®' 'ĸ'
feat  2218  loading -0.4719  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/2218]
  desc: specific color names or adjectives that describe color  | maxAct≈3.2, n_top_acts=63  | pos_logits: 'archive' ' stars' 'itage' 'edom' 'iliar'
feat 21222  loading -0.4632  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/21222]
  desc: numerical data and statistics within the text  | maxAct≈5.1, n_top_acts=64  | pos_logits: 'ĥ½' '·¸' 'Ĳ' 'č\n     ' '<|outofrange|>'
feat 29538  loading +0.4575  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/29538]
  desc: sequences of numbers and numerical patterns  | maxAct≈2.6, n_top_acts=55  | pos_logits: '_;' ':`' 'ensively' '/âĢĭ' 'cial'
feat 18090  loading -0.4522  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/18090]
  desc: scientific terms related to biology and medicine  | maxAct≈2.4, n_top_acts=64  | pos_logits: 'Ł' '£' '¢' 'Ń' 'µ'
feat  4730  loading +0.4445  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/4730]
  desc: programming keywords and syntax related to user interface actions and state management  | maxAct≈2.7, n_top_acts=58  | pos_logits: ' ()' '){#' '(_' ' <-' ' consists'
feat 14768  loading +0.4415  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/14768]
  desc: references to legal claims and judicial rulings  | maxAct≈1.5, n_top_acts=62  | pos_logits: ' (#' 'EVER' ' #' ' [#' ' [(\\['
feat  6467  loading +0.4403  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/6467]
  desc: terms and structures related to data management and software functions  | maxAct≈3.5, n_top_acts=65  | pos_logits: '=>' "':'" '©' 'Encoding' ' =>'
feat 10658  loading +0.4259  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/10658]
  desc: numeric values or quantities  | maxAct≈2.1, n_top_acts=63  | pos_logits: 'hip' '_;' 'Latin' '------' '=>'
feat  3173  loading -0.4240  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/3173]
  desc: numbers and numerical sequences  | maxAct≈4.7, n_top_acts=61  | pos_logits: 'istration' 'emon' 'ividual' 'illac' 'achines'

----------------------------------------------------------------------------------------------------
PC 6  (variance ratio = 0.0045, cumulative = 0.0548)
----------------------------------------------------------------------------------------------------
feat 28479  loading -0.4453  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/28479]
  desc: markup or structure in formatted text  | maxAct≈7.1, n_top_acts=52  | pos_logits: "'):" 'ubicin' '"){' 'osing' "':"
feat   531  loading -0.4174  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/531]
  desc: phrases related to programming components and functionality  | maxAct≈7.5, n_top_acts=62  | pos_logits: '"){' '![**' "']))" '"].' "')."
feat  7020  loading +0.4084  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/7020]
  desc: food-related terms and phrases that imply preparation or recipes  | maxAct≈1.5, n_top_acts=63  | pos_logits: ' itself' ' consists' ' comprises' ' contains' 'IPE'
feat 24696  loading -0.3882  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/24696]
  desc: syntactic structures involving programming commands and conditionals  | maxAct≈5.8, n_top_acts=65  | pos_logits: '_."' '\'"' 'nai' "']." 'ifice'
feat 10934  loading +0.3846  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/10934]
  desc: emotional and interpersonal themes, particularly those related to relationships and personal experiences  | maxAct≈4.1, n_top_acts=60  | pos_logits: '.[' '.' '.).' '.^[@' '^\\[[@'
feat 17535  loading -0.3785  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/17535]
  desc: proper names and identities of individuals  | maxAct≈2.3, n_top_acts=64  | pos_logits: ' respectively' 'æ°ı' 'à¯į' 'àµį' 'à±į'
feat 19894  loading -0.3680  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/19894]
  desc: details related to educational opportunities and qualifications  | maxAct≈3.5, n_top_acts=61  | pos_logits: ' others' 'ifice' ' other' ' manners' ' autres'
feat  5649  loading -0.3674  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/5649]
  desc: the use of conjunctions and repetitive phrases in sentences  | maxAct≈4.9, n_top_acts=67  | pos_logits: ' his' ' others' ' vice' ' other' ' remorse'
feat 25770  loading -0.3627  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/25770]
  desc: technical or programming-related terms and structures  | maxAct≈5.0, n_top_acts=60  | pos_logits: 'velopment' 'TRODUCTION' 'istration' 'isons' ' ashes'
feat 19846  loading -0.3620  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/19846]
  desc: phrases expressing emotions and curiosities about life experiences and relationships  | maxAct≈1.0, n_top_acts=59  | pos_logits: 'enny' '[/' '"));' 'yler' 'rax'

----------------------------------------------------------------------------------------------------
PC 7  (variance ratio = 0.0043, cumulative = 0.0590)
----------------------------------------------------------------------------------------------------
feat 20831  loading +0.3920  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/20831]
  desc: references to offering services and information for readers  | maxAct≈1.9, n_top_acts=60  | pos_logits: ' FREE' ' informative' ' updated' ' affiliate' ' newsletter'
feat 22786  loading +0.3820  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/22786]
  desc: references to document structures and technical specifications  | maxAct≈2.1, n_top_acts=61  | pos_logits: ' repositories' ' GitHub' ' Framework' 'BSD' ' binaries'
feat 18281  loading +0.3689  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/18281]
  desc: text related to software licensing and copyright information  | maxAct≈2.6, n_top_acts=58  | pos_logits: ' Redistributions' 'LICENSE' ' COPYRIGHT' ' copyright' ' licenses'
feat 12834  loading +0.3668  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/12834]
  desc: phrases related to data analysis and algorithm development  | maxAct≈1.0, n_top_acts=62  | pos_logits: 'npmjs' 'framework' ' incorporate' ' Framework' 'googleapis'
feat  9655  loading -0.3623  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/9655]
  desc: chemical compounds and their interactions relevant to biomedical research  | maxAct≈2.7, n_top_acts=55  | pos_logits: 'ĥ½' 'ģ' '·¸' 'Ĥ' 'ĳ'
feat  1102  loading +0.3615  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/1102]
  desc: references to digital content and tools related to various services and industries  | maxAct≈1.1, n_top_acts=58  | pos_logits: ' europÃ©' ' constitu' ' governance' ' parliamentary' ' reform'
feat  3869  loading +0.3610  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/3869]
  desc: references to software licenses and copyright information  | maxAct≈4.1, n_top_acts=39  | pos_logits: ' Redist' 'googleapis' ' Facts' ' Library' ' Catalog'
feat 15147  loading +0.3584  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/15147]
  desc: structured data and metadata related to API and resource definitions  | maxAct≈1.2, n_top_acts=58  | pos_logits: '://' ' default' ' config' 'ename' ' localhost'
feat 21375  loading +0.3581  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/21375]
  desc: references to URLs, particularly associated with GitHub and project repositories  | maxAct≈1.2, n_top_acts=53  | pos_logits: 'ĵ' '¾' 'į' 'µ' 'º'
feat 23725  loading +0.3538  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/23725]
  desc: identifiers and licensing information related to software  | maxAct≈5.4, n_top_acts=51  | pos_logits: ' License' ' copyrighted' 'pragma' ' license' ' compatible'

----------------------------------------------------------------------------------------------------
PC 8  (variance ratio = 0.0041, cumulative = 0.0631)
----------------------------------------------------------------------------------------------------
feat  6912  loading +0.4397  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/6912]
  desc: references to community and societal impacts across various regions and demographics  | maxAct≈2.3, n_top_acts=64  | pos_logits: ' including' ' intens' ' deserve' 'pective' ' area'
feat  9925  loading -0.4202  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/9925]
  desc: programming-related keywords and syntax  | maxAct≈2.2, n_top_acts=57  | pos_logits: 'Ģ' '¢' 'ħ' '²' 'İ'
feat 27344  loading -0.4186  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/27344]
  desc: programming-related actions and commands  | maxAct≈3.4, n_top_acts=52  | pos_logits: ' myself' 'Ļª' ' my' 'Ĳ' ' background'
feat 19722  loading -0.4124  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/19722]
  desc: actions related to commands or functions in programming contexts  | maxAct≈2.9, n_top_acts=60  | pos_logits: ' myself' ' them' ' my' ' aside' ' findViewById'
feat  2011  loading -0.4102  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/2011]
  desc: technical terms and concepts related to programming and functions  | maxAct≈1.5, n_top_acts=53  | pos_logits: '³' '¥' 'ľĵ' 'ķ' 'Ĵ'
feat 10034  loading -0.4011  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/10034]
  desc: keywords and components related to programming in the Microsoft .NET framework  | maxAct≈1.7, n_top_acts=63  | pos_logits: 'ª' 'ī' '¦' 'Ń' '³'
feat  6575  loading -0.3959  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/6575]
  desc: technical terms and keywords related to computer programming and systems  | maxAct≈1.2, n_top_acts=56  | pos_logits: ' etc' 'bytes' '.",' ' cursor' 'charts'
feat  8312  loading -0.3954  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/8312]
  desc: conditional and temporal phrases  | maxAct≈2.7, n_top_acts=51  | pos_logits: ' meu' ' btn' ' login' ' onCreate' ' logged'
feat 10934  loading +0.3898  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/10934]
  desc: emotional and interpersonal themes, particularly those related to relationships and personal experiences  | maxAct≈4.1, n_top_acts=60  | pos_logits: '.[' '.' '.).' '.^[@' '^\\[[@'
feat 28956  loading +0.3799  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/28956]
  desc: terms related to medical conditions and health issues, especially those associated with heart and metabolic diseases  | maxAct≈3.8, n_top_acts=55  | pos_logits: ' diagnosis' ' etiology' ' symptoms' ' prevention' ' disease'

----------------------------------------------------------------------------------------------------
PC 9  (variance ratio = 0.0039, cumulative = 0.0670)
----------------------------------------------------------------------------------------------------
feat 13156  loading -0.4414  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/13156]
  desc: statements related to medical research and findings  | maxAct≈4.0, n_top_acts=57  | pos_logits: 'ł' '¯' '»' ' Alternatively' ' METHODS'
feat 19170  loading +0.4137  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/19170]
  desc: references to locations, particularly community events and organizations  | maxAct≈2.2, n_top_acts=60  | pos_logits: ' Studios' 'ville' ' Magazine' ' Gallery' ' Creek'
feat  7869  loading +0.3861  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/7869]
  desc: specific geographic locations or landmarks  | maxAct≈3.6, n_top_acts=63  | pos_logits: 'stown' ' area' ' vicinity' ' subdivision' ' Basin'
feat   564  loading -0.3858  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/564]
  desc: descriptions of inventions or technologies related to power supply and sensor systems  | maxAct≈3.8, n_top_acts=53  | pos_logits: ' Alternatively' ' Thereafter' ' Patent' ' Accordingly' ' Further'
feat 13571  loading -0.3850  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/13571]
  desc: phrases related to medical treatment and conditions  | maxAct≈4.0, n_top_acts=60  | pos_logits: '»¿' 'ľĵ' 'ĨĴ' 'Ļª' 'Ĳ'
feat  6038  loading -0.3804  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/6038]
  desc: references to statistical or numerical data in the text  | maxAct≈4.7, n_top_acts=57  | pos_logits: '^[@' ' Briefly' '[@' ' ^[@' '\\[[@'
feat 10469  loading -0.3774  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/10469]
  desc: connections between physical health factors and their potential effects on cognitive and behavioral outcomes  | maxAct≈1.7, n_top_acts=59  | pos_logits: ' influencing' ' effects' ' factors' ' affect' ' affecting'
feat 31880  loading +0.3585  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/31880]
  desc: phrases indicating international reach or presence  | maxAct≈2.3, n_top_acts=59  | pos_logits: ' industry' 'indust' ' country' ' clubs' ' continents'
feat 21674  loading +0.3540  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/21674]
  desc: combinations of product names and their characteristics  | maxAct≈2.2, n_top_acts=61  | pos_logits: ' Productions' ' Championships' ' Championship' ' Pictures' ' Products'
feat 17200  loading -0.3514  [https://www.neuronpedia.org/pythia-70m-deduped/3-res-sm/17200]
  desc: the presence of specific nouns and structural terminology related to mechanical or electronic components  | maxAct≈4.1, n_top_acts=58  | pos_logits: ' foregoing' ' aforementioned' '»¿' ' latter' 'ĻĤ'

####################################################################################################
# res-sm L0  (pythia-70m-deduped-res-sm / blocks.0.hook_resid_post)
####################################################################################################

n_features=32,768, d_model=512
Participation ratio: 296.0  |  top 10 PC variance ratios: ['0.0324', '0.0152', '0.0099', '0.0082', '0.0061', '0.0057', '0.0054', '0.0048', '0.0045', '0.0044']
Cumulative variance of top 10: 0.0967

----------------------------------------------------------------------------------------------------
PC 0  (variance ratio = 0.0324, cumulative = 0.0324)
----------------------------------------------------------------------------------------------------
feat 10361  loading +0.7739  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/10361]
  desc: occurrences of coding syntax and structure in programming languages  | maxAct≈0.7, n_top_acts=56  | pos_logits: 'ĥ½' 'Ĺ' 'ľ' '¿½' 'ãģĦ'
feat 20809  loading +0.7577  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/20809]
  desc: various elements related to HTML or XML syntax  | maxAct≈0.6, n_top_acts=57  | pos_logits: 'ÑĢÐ¾Ñģ' 'ĥ½' ">';" 'ÑĢÑĥ' 'Ñı'
feat 19276  loading +0.7439  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/19276]
  desc: elements related to HTML and web programming code  | maxAct≈0.9, n_top_acts=61  | pos_logits: 'ĥ½' '<|outofrange|>' '               ' '                                             ' '\n\n                   '
feat 10154  loading +0.6968  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/10154]
  desc: keywords and phrases indicating significant financial or economic concepts  | maxAct≈4.9, n_top_acts=69  | pos_logits: '  \n' ' blame' ' sudden' 'äºİ' " '')"
feat 27252  loading +0.6695  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/27252]
  desc: structured data formats and metadata representations  | maxAct≈0.4, n_top_acts=65  | pos_logits: 'woke' ' calm' 'ITED' ' ÑĩÑĤÐ¾' ']>'
feat 24435  loading +0.6462  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/24435]
  desc: expressions of satisfaction or recommendation  | maxAct≈2.9, n_top_acts=68  | pos_logits: 'Ĺ' 'Ĥ' ' finding' ' heart' '¨'
feat 21482  loading +0.6412  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/21482]
  desc: mathematical symbols and punctuation  | maxAct≈1.3, n_top_acts=63  | pos_logits: 'rapeutic' ' finding' '![](' ' descent' 'OSE'
feat 17241  loading +0.6275  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/17241]
  desc: legal terminology and references related to laws and regulations  | maxAct≈0.6, n_top_acts=62  | pos_logits: ' nothing' 'Īĺ' '¿½' 'ĥ½' ' truth'
feat 27107  loading +0.6254  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/27107]
  desc: information about medical treatments and their legal implications  | maxAct≈0.9, n_top_acts=60  | pos_logits: ' faint' ' descent' 'urred' ' difference' ' steep'
feat 19152  loading +0.6204  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/19152]
  desc: mathematical notation and symbols in formal content  | maxAct≈0.8, n_top_acts=59  | pos_logits: 'urane' 'Ã§Ã£o' 'ansen' 'Ñī' 'ĥ½'

----------------------------------------------------------------------------------------------------
PC 1  (variance ratio = 0.0152, cumulative = 0.0476)
----------------------------------------------------------------------------------------------------
feat  4900  loading -0.5837  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/4900]
  desc: dependencies and relationships within sentences  | maxAct≈3.8, n_top_acts=58  | pos_logits: 'ĻĤ' 'Ļª' 'ĨĴ' 'ľĵ' 'ĸ´'
feat 10952  loading -0.5319  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/10952]
  desc: instances of personal pronouns and their usage in sentences  | maxAct≈0.8, n_top_acts=58  | pos_logits: 'ĻĤ' 'Ļª' '»¿' ' simplest' ' nothing'
feat 28011  loading -0.4985  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/28011]
  desc: mentions of comprehensive plans or approaches  | maxAct≈6.0, n_top_acts=45  | pos_logits: '¸' '¿½' 'ĻĤ' 'ķ' 'Ĥ'
feat  2375  loading -0.4926  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/2375]
  desc: terms related to improvement or enhancement  | maxAct≈5.5, n_top_acts=53  | pos_logits: ' detection' ' efficiency' 'ł' 'ively' ' transparency'
feat  2474  loading +0.4881  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/2474]
  desc: structured data formats or elements related to programming and development  | maxAct≈1.6, n_top_acts=59  | pos_logits: 'iÃ¨re' ' yourselves' ' itself' ' yourself' 'nement'
feat  3235  loading -0.4866  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/3235]
  desc: references to agendas and hierarchical structures in political or ideological contexts  | maxAct≈2.1, n_top_acts=61  | pos_logits: 'STEM' '¹' 'Ĺ' '°' 'xture'
feat 10456  loading -0.4847  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/10456]
  desc: references to legislative and regulatory processes  | maxAct≈2.8, n_top_acts=64  | pos_logits: ' regulating' 'ctor' 'oke' 'rust' 'ulate'
feat 27693  loading +0.4812  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/27693]
  desc: mentions of specific companies or organizations involved in affiliate advertising  | maxAct≈1.0, n_top_acts=56  | pos_logits: ' Ã¨' 'stein' 'liche' ' (,' 'bp'
feat 21068  loading -0.4797  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/21068]
  desc: references to adequacy or the lack thereof in various contexts  | maxAct≈6.5, n_top_acts=40  | pos_logits: 'ĻĤ' 'º' 'ļ' 'Ł' 'ħ'
feat 16190  loading -0.4752  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/16190]
  desc: references to various technologies  | maxAct≈5.7, n_top_acts=40  | pos_logits: 'urgical' 'cles' 'ector' 'mith' 'igner'

----------------------------------------------------------------------------------------------------
PC 2  (variance ratio = 0.0099, cumulative = 0.0576)
----------------------------------------------------------------------------------------------------
feat 18629  loading -0.4902  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/18629]
  desc: references to the letter "K" with varying importance  | maxAct≈6.3, n_top_acts=39  | pos_logits: '[^' '&' '!_' '**\\' '!'
feat  7819  loading -0.4887  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/7819]
  desc: references to specific characters or entities  | maxAct≈6.1, n_top_acts=36  | pos_logits: 'weet' ' ranked' 'WE' ' rated' 'âĢĿâĢĶ'
feat  4862  loading -0.4728  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/4862]
  desc: references to specific culinary or food-related concepts  | maxAct≈5.8, n_top_acts=37  | pos_logits: 'UST' 'redited' 'ube' 'otted' ' screened'
feat  2521  loading -0.4721  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/2521]
  desc: syntactical structures and punctuation in code-like text  | maxAct≈0.5, n_top_acts=55  | pos_logits: 'pal' 'net' 'pieces' 'Ħ' 'ģ'
feat  2181  loading -0.4689  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/2181]
  desc: the repeated occurrence of the letter "S"  | maxAct≈6.0, n_top_acts=39  | pos_logits: 'ubl' '![' 'ellers' 'BT' 'eller'
feat 32196  loading -0.4664  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/32196]
  desc: the presence of a specific term related to biological classifications or taxonomies  | maxAct≈6.5, n_top_acts=40  | pos_logits: 'REEK' 'Simplify' '![' ' induct' 'âĢĿâĢĶ'
feat 29055  loading -0.4649  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/29055]
  desc: references to legal case codes or identifiers  | maxAct≈6.1, n_top_acts=40  | pos_logits: 'ateral' 'ambda' 'ian' 'attice' 'MSC'
feat 31262  loading -0.4594  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/31262]
  desc: the repetition of the letter "H" followed by a digit  | maxAct≈6.5, n_top_acts=34  | pos_logits: ' pylori' 'ock' 'SB' 'ilde' 'WE'
feat 13242  loading -0.4535  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/13242]
  desc: references to specific geographic locations or entities  | maxAct≈6.5, n_top_acts=41  | pos_logits: 'ipline' ' screened' 'ingly' ' critically' '**'
feat 28316  loading -0.4459  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/28316]
  desc: the repetition of the letter 'B' in various contexts  | maxAct≈5.7, n_top_acts=34  | pos_logits: 'acillus' 'icultural' 'less' 'ILITY' 'ECs'

----------------------------------------------------------------------------------------------------
PC 3  (variance ratio = 0.0082, cumulative = 0.0657)
----------------------------------------------------------------------------------------------------
feat  1730  loading -0.4233  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/1730]
  desc: terms related to marine biology and environmental elements  | maxAct≈1.9, n_top_acts=60  | pos_logits: 'heet' 'ystems' 'pace' 'helf' 'heets'
feat 21743  loading +0.3970  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/21743]
  desc: specific characters or symbols indicative of unusual formatting or encoding errors  | maxAct≈0.7, n_top_acts=60  | pos_logits: '                     ' '<|outofrange|>' '\n        ' '\n\n\n     ' '\n                            '
feat 29168  loading +0.3857  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/29168]
  desc: specific names and references related to people and places  | maxAct≈2.3, n_top_acts=60  | pos_logits: 'uer' 'thor' 'uen' ' clicked' 'chers'
feat 29224  loading -0.3745  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/29224]
  desc: references to abstract concepts and structured ideas  | maxAct≈1.2, n_top_acts=61  | pos_logits: 'makers' 'iners' 'Metrics' 'akers' 'ellers'
feat 28390  loading +0.3731  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/28390]
  desc: terms related to technological components and communication systems  | maxAct≈0.9, n_top_acts=67  | pos_logits: ' legend' 'stock' ' purchased' 'ieux' 'enstein'
feat  3101  loading -0.3716  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/3101]
  desc: references to customer service and operational aspects of businesses  | maxAct≈0.9, n_top_acts=60  | pos_logits: 'pieces' 'wiki' 'ptions' 'citations' ' (Â§'
feat  3085  loading +0.3710  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/3085]
  desc: legal terms and structures typically used in court and legal documents  | maxAct≈0.7, n_top_acts=61  | pos_logits: '\n\n                   ' '\n                                                                ' 'č\n        ' 'č\n      ' '                                                                               '
feat  3451  loading +0.3708  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/3451]
  desc: verbs related to unveiling or revealing something new  | maxAct≈2.7, n_top_acts=66  | pos_logits: '½' '¨' '¯' '·' '¤'
feat 14195  loading +0.3704  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/14195]
  desc: brand names and product identifiers related to technology  | maxAct≈1.4, n_top_acts=57  | pos_logits: ' ' ' ' '\n\n            ' ' \n               ' '                       '
feat 24338  loading -0.3704  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/24338]
  desc: terms related to therapy and medical treatments  | maxAct≈5.0, n_top_acts=30  | pos_logits: '"}](#' 'ska' 'genstein' 'garten' 'aho'

----------------------------------------------------------------------------------------------------
PC 4  (variance ratio = 0.0061, cumulative = 0.0718)
----------------------------------------------------------------------------------------------------
feat  8590  loading +0.4171  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/8590]
  desc: (no auto-interp description)  | maxAct≈0.0, n_top_acts=0  | pos_logits: 
feat 26471  loading +0.3858  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/26471]
  desc: terms related to cellular structures and functions  | maxAct≈7.6, n_top_acts=25  | pos_logits: 'uary' 'imedia' '¸' '¬' 'ĻĤ'
feat 12014  loading +0.3805  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/12014]
  desc: medical terminologies related to anatomical structures and conditions  | maxAct≈5.8, n_top_acts=25  | pos_logits: 'istics' 'ruptcy' ' deserves' 'itud' '-âĤ¬'
feat 28788  loading +0.3751  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/28788]
  desc: terms related to specific types of blood cell conditions or abnormalities  | maxAct≈7.2, n_top_acts=25  | pos_logits: '£' '¡' 'Ĵ' 'Ģ' 'Ł'
feat  5088  loading +0.3720  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/5088]
  desc: medical terms related to throat and oral anatomy, particularly focusing on cancers and conditions affecting those areas  | maxAct≈6.9, n_top_acts=25  | pos_logits: 'issance' 'caster' 'istor' 'Åĳ' 'view'
feat  3576  loading +0.3713  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/3576]
  desc: terms related to health and safety in medical and biological contexts  | maxAct≈0.7, n_top_acts=52  | pos_logits: 'ness' 'thouse' 'vity' 'istics' 'ista'
feat  2073  loading +0.3708  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/2073]
  desc: terms related to cellular biology  | maxAct≈1.5, n_top_acts=30  | pos_logits: 'Ł' '¡' 'Ģ' 'ł' 'ŀ'
feat  9219  loading +0.3691  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/9219]
  desc: terms related to atmospheric and environmental conditions  | maxAct≈3.7, n_top_acts=51  | pos_logits: '¤' '¦' '£' 'Ł' '»¿'
feat 11289  loading +0.3678  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/11289]
  desc: words associated with infectious diseases and the concept of violence or aggression  | maxAct≈7.0, n_top_acts=27  | pos_logits: 'ness' 'ulence' 'apine' 'issance' 'liest'
feat 21923  loading +0.3663  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/21923]
  desc: references to artistic and architectural concepts  | maxAct≈2.8, n_top_acts=66  | pos_logits: 'µ' 'ĸ' 'ĻĤ' 'Ĥ' '£'

----------------------------------------------------------------------------------------------------
PC 5  (variance ratio = 0.0057, cumulative = 0.0775)
----------------------------------------------------------------------------------------------------
feat  9605  loading -0.4288  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/9605]
  desc: terms related to academic semesters and course structures  | maxAct≈2.2, n_top_acts=66  | pos_logits: 'ĥ½' 'Ĵ' 'ī' 'vi' 'otyping'
feat 13555  loading +0.4100  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/13555]
  desc: instances of the word "Washington."  | maxAct≈7.4, n_top_acts=30  | pos_logits: 'àµį' 'àµ' ' tone' 'ICAN' 'venue'
feat 18332  loading +0.4099  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/18332]
  desc: mentions of Australia and Australian-related topics  | maxAct≈6.8, n_top_acts=42  | pos_logits: 'selves' 'àµį' 'toolt' 'izumab' 'duty'
feat 23945  loading +0.4050  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/23945]
  desc: terms and concepts related to drug treatment and medications  | maxAct≈2.2, n_top_acts=59  | pos_logits: ' Order' ' Council' ' Utility' ' Improvement' ' Week'
feat  4278  loading +0.4028  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/4278]
  desc: technical terms related to analysis, measurement, and performance in various fields  | maxAct≈2.0, n_top_acts=63  | pos_logits: ' League' ' Engineer' ' Corps' ' Manager' ' Studios'
feat 13086  loading +0.3998  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/13086]
  desc: references to annual events and meetings  | maxAct≈2.2, n_top_acts=61  | pos_logits: ' Editor' ' Manager' ' League' 'fileID' ')$)'
feat 19207  loading +0.3942  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/19207]
  desc: references to America and its citizens  | maxAct≈6.1, n_top_acts=54  | pos_logits: ' Affairs' 'skins' 'eLife' 'illac' 'iast'
feat 27706  loading +0.3925  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/27706]
  desc: references to Scotland and its related entities  | maxAct≈6.3, n_top_acts=36  | pos_logits: 'bourg' 'herry' 'boro' 'borough' 'bourne'
feat  6166  loading +0.3908  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/6166]
  desc: references to European nations or contexts  | maxAct≈6.5, n_top_acts=42  | pos_logits: 'herry' 'eLife' 'view' 'iator' 'noreply'
feat 10432  loading +0.3895  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/10432]
  desc: references to the province of Ontario and related geographical locations  | maxAct≈6.3, n_top_acts=45  | pos_logits: "'s" 'EXPORT' ' herself' 'leen' 'ville'

----------------------------------------------------------------------------------------------------
PC 6  (variance ratio = 0.0054, cumulative = 0.0829)
----------------------------------------------------------------------------------------------------
feat 16406  loading +0.3824  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/16406]
  desc: terms related to health treatments and their documentation  | maxAct≈1.9, n_top_acts=60  | pos_logits: 'quier' 'icine' 'ness' 'ictionary' 'eca'
feat 20391  loading +0.3727  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/20391]
  desc: references to groups and their interactions in a programming context  | maxAct≈1.9, n_top_acts=67  | pos_logits: 'Kit' ' Development' 'zilla' 'bridge' 'lickr'
feat 16477  loading +0.3716  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/16477]
  desc: terms and concepts related to film industry roles and characteristics  | maxAct≈1.4, n_top_acts=64  | pos_logits: 'doll' 'blogger' 'ifornia' 'cott' 'quote'
feat 12154  loading +0.3687  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/12154]
  desc: commands and user interface elements related to visibility and interaction controls  | maxAct≈2.2, n_top_acts=60  | pos_logits: 'blogger' 'aurus' 'ful' 'ivent' 'cule'
feat  2843  loading +0.3610  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/2843]
  desc: technical terms related to frequency and speed  | maxAct≈2.6, n_top_acts=60  | pos_logits: 'nement' 'ings' 'iento' 'charts' 'naire'
feat 31180  loading +0.3516  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/31180]
  desc: references to rendering components and configurations in a graphical context  | maxAct≈2.5, n_top_acts=67  | pos_logits: ' Agric' ' Authority' 'ERTY' ' Caption' ')|$('
feat 14970  loading +0.3434  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/14970]
  desc: words related to API requests and responses  | maxAct≈2.2, n_top_acts=69  | pos_logits: 'ħ' 'naire' 'istas' 'blogger' ' arsenal'
feat 21044  loading +0.3370  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/21044]
  desc: references to active status or activity-related terms  | maxAct≈5.2, n_top_acts=43  | pos_logits: 'ivism' ' ingredient' 'mind' 'quarters' 'etics'
feat 24312  loading +0.3319  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/24312]
  desc: terms related to programming constructs and data structures  | maxAct≈1.8, n_top_acts=61  | pos_logits: 'zilla' ' Publishing' 'bank' 'journals' '¾'
feat 32585  loading +0.3292  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/32585]
  desc: references to real estate-related tasks and actions  | maxAct≈2.1, n_top_acts=69  | pos_logits: 'ħ' '¼' 'Verlag' 'ł' 'Ł'

----------------------------------------------------------------------------------------------------
PC 7  (variance ratio = 0.0048, cumulative = 0.0877)
----------------------------------------------------------------------------------------------------
feat 24138  loading +0.3388  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/24138]
  desc: XML document declarations and related tags  | maxAct≈7.0, n_top_acts=35  | pos_logits: '\n\n    ' '\n  ' '\n         ' '                                                                          ' '\n                            '
feat 19037  loading +0.3192  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/19037]
  desc: references to software licenses  | maxAct≈8.6, n_top_acts=47  | pos_logits: 'lement' 'rowser' ' vacancy' 'bject' 'Ã©tÃ©'
feat  1398  loading -0.3145  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/1398]
  desc: references to time, specifically past time indicators  | maxAct≈5.5, n_top_acts=33  | pos_logits: 'engers' 'bels' 'eur' '³' 'ÅĦ'
feat 17831  loading +0.3062  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/17831]
  desc: HTML document type declarations  | maxAct≈8.4, n_top_acts=25  | pos_logits: 'Ã³l' ' Rptr' ' pist' 'ettes' 'rst'
feat 22075  loading -0.3018  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/22075]
  desc: the repeated occurrence of a specific name or term  | maxAct≈5.0, n_top_acts=43  | pos_logits: 'hurst' 'wic' 'ENTIAL' 'fficients' 'velt'
feat 26217  loading -0.3004  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/26217]
  desc: names or surnames of individuals, particularly those related to sports or public figures  | maxAct≈2.6, n_top_acts=66  | pos_logits: 'ÅĦ' 'pora' 'vist' 'stown' 'recht'
feat 18283  loading +0.2985  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/18283]
  desc: (no auto-interp description)  | maxAct≈0.0, n_top_acts=0  | pos_logits: 
feat 32329  loading +0.2965  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/32329]
  desc: references to software licenses  | maxAct≈7.1, n_top_acts=33  | pos_logits: 'eer' 'sten' 'imento' 'taire' ' profession'
feat 29463  loading +0.2964  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/29463]
  desc: instances of the word "abstract" in various contexts  | maxAct≈2.8, n_top_acts=28  | pos_logits: 'ictionary' 'ividual' ' aloud' 'ruary' ' dissertation'
feat   401  loading +0.2962  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/401]
  desc: image file formats  | maxAct≈6.5, n_top_acts=25  | pos_logits: 'eness' 'itzer' 'ictional' 'inished' 'msgstr'

----------------------------------------------------------------------------------------------------
PC 8  (variance ratio = 0.0045, cumulative = 0.0922)
----------------------------------------------------------------------------------------------------
feat 17441  loading -0.3578  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/17441]
  desc: terms related to health-related interventions and initiatives  | maxAct≈5.7, n_top_acts=29  | pos_logits: 'orbent' 'ister' ' targeting' 'isters' ' oxidase'
feat  4546  loading -0.3374  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/4546]
  desc: references to side effects of drugs and treatments  | maxAct≈5.8, n_top_acts=47  | pos_logits: 'pace' 'uate' 'ystem' 'ball' 'cript'
feat 18413  loading -0.3363  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/18413]
  desc: text that contains references to arguments and claims in a debate or discussion context  | maxAct≈2.5, n_top_acts=64  | pos_logits: 'heet' 'hell' 'creen' 'erver' 'helf'
feat  4408  loading -0.3363  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/4408]
  desc: the occurrence of the word "insult" in various contexts  | maxAct≈6.2, n_top_acts=43  | pos_logits: 'ulin' ' Forces' 'pective' 'ulator' ' Commissioner'
feat  2013  loading -0.3223  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/2013]
  desc: terminology related to research methodologies and efficacy evaluations  | maxAct≈3.8, n_top_acts=27  | pos_logits: 'hift' 'uate' 'áĢº' 'à«ĩ' 'eer'
feat 20106  loading -0.3198  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/20106]
  desc: references to interactions, particularly in a scientific or biological context  | maxAct≈6.0, n_top_acts=34  | pos_logits: 'pace' 'frame' 'controller' 'creen' 'Controller'
feat 11085  loading -0.3171  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/11085]
  desc: references to emotional or physiological reactions  | maxAct≈4.6, n_top_acts=36  | pos_logits: 'ystem' 'ystems' ' occurring' 'ymbol' 'ome'
feat  1778  loading -0.3147  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/1778]
  desc: references to various approaches or methodologies for problem-solving or optimization  | maxAct≈5.7, n_top_acts=44  | pos_logits: 'urgical' 'hell' 'ione' 'iph' 'urg'
feat 11679  loading -0.3139  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/11679]
  desc: occurrences of the word "actions" and its variations  | maxAct≈4.3, n_top_acts=53  | pos_logits: 'cript' 'ional' 'naire' 'ense' 'ary'
feat  9654  loading -0.3106  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/9654]
  desc: references to issues or difficulties in various contexts  | maxAct≈5.6, n_top_acts=45  | pos_logits: ' encountered' 'uit' ' faced' 'igue' 'ymmetric'

----------------------------------------------------------------------------------------------------
PC 9  (variance ratio = 0.0044, cumulative = 0.0967)
----------------------------------------------------------------------------------------------------
feat 28877  loading +0.3412  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/28877]
  desc: the frequency of the term "Center" in various contexts  | maxAct≈5.7, n_top_acts=49  | pos_logits: 'piece' 'pieces' 'yard' 'lif' 'strom'
feat  1528  loading +0.3322  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/1528]
  desc: references to community organizations and centers  | maxAct≈6.2, n_top_acts=33  | pos_logits: 'ycin' 'ville' 'holder' 'pieces' 'holm'
feat  5815  loading +0.3165  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/5815]
  desc: references to regulatory bodies or commissions  | maxAct≈5.7, n_top_acts=50  | pos_logits: 'ual' 'ery' 'isiÃ³n' 'etable' 'esy'
feat 29330  loading -0.3163  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/29330]
  desc: references to movement or actions  | maxAct≈2.6, n_top_acts=69  | pos_logits: 'mith' 'pace' 'ource' 'heet' 'chool'
feat 21085  loading -0.3140  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/21085]
  desc: references to emotional connections and the concept of "hearts."  | maxAct≈2.6, n_top_acts=57  | pos_logits: 'hip' 'helf' 'pace' 'hell' 'ymbol'
feat 29884  loading -0.3136  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/29884]
  desc: instances or references to smiling and positive facial expressions  | maxAct≈4.2, n_top_acts=69  | pos_logits: ' hello' 'backed' 'uating' 'PING' 'rpc'
feat 11427  loading -0.3131  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/11427]
  desc: mentions of gases and fluids, particularly in environmental and medical contexts  | maxAct≈2.3, n_top_acts=61  | pos_logits: 'creen' 'ystems' 'heet' 'mith' 'pace'
feat  4848  loading +0.3076  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/4848]
  desc: references to "Central" locations  | maxAct≈6.6, n_top_acts=42  | pos_logits: 'itat' 'istani' 'ifornia' 'ÃŃa' 'esar'
feat 26281  loading +0.3036  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/26281]
  desc: references to community and related concepts  | maxAct≈6.1, n_top_acts=38  | pos_logits: 'wide' "'s" 'ulence' 'view' ' estimates'
feat 26335  loading +0.2974  [https://www.neuronpedia.org/pythia-70m-deduped/0-res-sm/26335]
  desc: references to specific geographic regions  | maxAct≈5.9, n_top_acts=35  | pos_logits: 'ally' 'ielle' 'iale' 'wide' 'ize'

====================================================================================================
SUMMARY: what each layer's geometric structure is 'about'
====================================================================================================
(Fill in interpretation manually after reading the per-layer PCs above.)

  mlp-sm L5     PR= 163.0  top1=0.0598  cum10=0.142
  res-sm L3     PR= 422.7  top1=0.0144  cum10=0.067
  res-sm L0     PR= 296.0  top1=0.0324  cum10=0.097

