from PipelineFinder import PipelineSelection as PS

""" Possible data sets:

Baseline: Normal patients (0), Non-Lesional patients (1), Lesional (2)

Psoriasis              GSE13355          180         NN = Normal, PN = Non-Lesional, PP = Lesional
                       GSE30999          170         No normal patients
                       GSE34248          28          No normal patients
                       GSE41662          48          No normal patients
                       GSE78097          33          Different: Normal (0), Mild (1), Severe Psoriasis (2)
                       GSE14905          82                  
Atopic  dermatitis     GSE32924          33                  
                       GSE27887          35          Different: Pre NL (0), Post NL (1), Pre L (2), Post L (3)
                       GSE36842          39          Also tested difference between Acute (2) and Chronic (3) Dermatitis

"""

PS.select_pipeline_tpot('GSE13355', train_size=0.9, max_opt_time=120, n_gen=100, pop_size=10)

PS.select_pipeline_tpot('GSE30999', train_size=0.9, max_opt_time=120, n_gen=100, pop_size=10)

PS.select_pipeline_tpot('GSE34248', train_size=0.9, max_opt_time=120, n_gen=100, pop_size=10)

PS.select_pipeline_tpot('GSE41662', train_size=0.9, max_opt_time=120, n_gen=100, pop_size=10)

PS.select_pipeline_tpot('GSE78097', train_size=0.9, max_opt_time=120, n_gen=100, pop_size=10)

PS.select_pipeline_tpot('GSE14905', train_size=0.9, max_opt_time=120, n_gen=100, pop_size=10)

PS.select_pipeline_tpot('GSE32924', train_size=0.9, max_opt_time=120, n_gen=100, pop_size=10)

PS.select_pipeline_tpot('GSE27887', train_size=0.9, max_opt_time=120, n_gen=100, pop_size=10)

PS.select_pipeline_tpot('GSE36842', train_size=0.9, max_opt_time=120, n_gen=100, pop_size=10)