"""Script for parsing emerson data without unzipping it"""
import pandas as pd
import numpy as np
import os, sys
import zipfile as zf

#Stuff I could've probly sent you the pickles but it doesn't matter you can just run the script as it is
HLA_A= np.array(['A01', 'A02', 'A03', 'A11', 'A23', 'A24', 'A25', 'A26', 'A28',
       'A29', 'A30', 'A31', 'A32', 'A33', 'A34', 'A36', 'A66', 'A68',
       'A69', 'A74', 'A80'], dtype='<U7')
HLA_B = np.array(['B05', 'B07', 'B08', 'B13', 'B14', 'B15', 'B17', 'B18', 'B22',
       'B27', 'B35', 'B37', 'B38', 'B39', 'B40', 'B41', 'B42', 'B44',
       'B45', 'B47', 'B48', 'B49', 'B50', 'B51', 'B52', 'B53', 'B54',
       'B55', 'B56', 'B57', 'B58', 'B60', 'B61', 'B62', 'B63', 'B70',
       'B73', 'B75', 'B78', 'B82'], dtype='<U7')
names = np.array(['HIP00110.tsv', 'HIP00169.tsv', 'HIP00594.tsv', 'HIP00602.tsv',
       'HIP00614.tsv', 'HIP00640.tsv', 'HIP00707.tsv', 'HIP00710.tsv',
       'HIP00715.tsv', 'HIP00728.tsv', 'HIP00734.tsv', 'HIP00761.tsv',
       'HIP00769.tsv', 'HIP00771.tsv', 'HIP00773.tsv', 'HIP00775.tsv',
       'HIP00777.tsv', 'HIP00779.tsv', 'HIP00805.tsv', 'HIP00813.tsv',
       'HIP00819.tsv', 'HIP00822.tsv', 'HIP00825.tsv', 'HIP00826.tsv',
       'HIP00832.tsv', 'HIP00838.tsv', 'HIP00851.tsv', 'HIP00869.tsv',
       'HIP00898.tsv', 'HIP00904.tsv', 'HIP00924.tsv', 'HIP00926.tsv',
       'HIP00934.tsv', 'HIP00951.tsv', 'HIP00985.tsv', 'HIP00997.tsv',
       'HIP00999.tsv', 'HIP01004.tsv', 'HIP01022.tsv', 'HIP01055.tsv',
       'HIP01129.tsv', 'HIP01140.tsv', 'HIP01160.tsv', 'HIP01162.tsv',
       'HIP01180.tsv', 'HIP01181.tsv', 'HIP01197.tsv', 'HIP01206.tsv',
       'HIP01218.tsv', 'HIP01220.tsv', 'HIP01223.tsv', 'HIP01232.tsv',
       'HIP01255.tsv', 'HIP01264.tsv', 'HIP01266.tsv', 'HIP01298.tsv',
       'HIP01313.tsv', 'HIP01359.tsv', 'HIP01384.tsv', 'HIP01391.tsv',
       'HIP01392.tsv', 'HIP01393.tsv', 'HIP01465.tsv', 'HIP01470.tsv',
       'HIP01499.tsv', 'HIP01501.tsv', 'HIP01571.tsv', 'HIP01582.tsv',
       'HIP01795.tsv', 'HIP01797.tsv', 'HIP01798.tsv', 'HIP01805.tsv',
       'HIP01820.tsv', 'HIP01850.tsv', 'HIP01856.tsv', 'HIP01865.tsv',
       'HIP01867.tsv', 'HIP01870.tsv', 'HIP01947.tsv', 'HIP02024.tsv',
       'HIP02078.tsv', 'HIP02090.tsv', 'HIP02103.tsv', 'HIP02112.tsv',
       'HIP02126.tsv', 'HIP02663.tsv', 'HIP02734.tsv', 'HIP02737.tsv',
       'HIP02742.tsv', 'HIP02805.tsv', 'HIP02811.tsv', 'HIP02820.tsv',
       'HIP02848.tsv', 'HIP02855.tsv', 'HIP02873.tsv', 'HIP02875.tsv',
       'HIP02877.tsv', 'HIP02928.tsv', 'HIP02931.tsv', 'HIP02947.tsv',
       'HIP02962.tsv', 'HIP02997.tsv', 'HIP03004.tsv', 'HIP03099.tsv',
       'HIP03107.tsv', 'HIP03111.tsv', 'HIP03184.tsv', 'HIP03228.tsv',
       'HIP03233.tsv', 'HIP03236.tsv', 'HIP03275.tsv', 'HIP03370.tsv',
       'HIP03378.tsv', 'HIP03381.tsv', 'HIP03383.tsv', 'HIP03385.tsv',
       'HIP03484.tsv', 'HIP03494.tsv', 'HIP03495.tsv', 'HIP03502.tsv',
       'HIP03505.tsv', 'HIP03511.tsv', 'HIP03591.tsv', 'HIP03592.tsv',
       'HIP03597.tsv', 'HIP03618.tsv', 'HIP03628.tsv', 'HIP03630.tsv',
       'HIP03651.tsv', 'HIP03677.tsv', 'HIP03678.tsv', 'HIP03685.tsv',
       'HIP03693.tsv', 'HIP03695.tsv', 'HIP03720.tsv', 'HIP03732.tsv',
       'HIP03807.tsv', 'HIP03812.tsv', 'HIP04455.tsv', 'HIP04464.tsv',
       'HIP04471.tsv', 'HIP04475.tsv', 'HIP04480.tsv', 'HIP04498.tsv',
       'HIP04509.tsv', 'HIP04510.tsv', 'HIP04511.tsv', 'HIP04527.tsv',
       'HIP04532.tsv', 'HIP04545.tsv', 'HIP04552.tsv', 'HIP04555.tsv',
       'HIP04576.tsv', 'HIP04578.tsv', 'HIP04597.tsv', 'HIP04605.tsv',
       'HIP04611.tsv', 'HIP04634.tsv', 'HIP04958.tsv', 'HIP05311.tsv',
       'HIP05331.tsv', 'HIP05377.tsv', 'HIP05388.tsv', 'HIP05390.tsv',
       'HIP05398.tsv', 'HIP05405.tsv', 'HIP05409.tsv', 'HIP05434.tsv',
       'HIP05437.tsv', 'HIP05444.tsv', 'HIP05455.tsv', 'HIP05460.tsv',
       'HIP05467.tsv', 'HIP05524.tsv', 'HIP05533.tsv', 'HIP05535.tsv',
       'HIP05540.tsv', 'HIP05551.tsv', 'HIP05552.tsv', 'HIP05559.tsv',
       'HIP05561.tsv', 'HIP05563.tsv', 'HIP05574.tsv', 'HIP05578.tsv',
       'HIP05590.tsv', 'HIP05595.tsv', 'HIP05665.tsv', 'HIP05757.tsv',
       'HIP05763.tsv', 'HIP05815.tsv', 'HIP05817.tsv', 'HIP05832.tsv',
       'HIP05838.tsv', 'HIP05841.tsv', 'HIP05934.tsv', 'HIP05941.tsv',
       'HIP05942.tsv', 'HIP05948.tsv', 'HIP05960.tsv', 'HIP06191.tsv',
       'HIP07754.tsv', 'HIP08076.tsv', 'HIP08200.tsv', 'HIP08223.tsv',
       'HIP08230.tsv', 'HIP08236.tsv', 'HIP08305.tsv', 'HIP08337.tsv',
       'HIP08339.tsv', 'HIP08345.tsv', 'HIP08346.tsv', 'HIP08389.tsv',
       'HIP08399.tsv', 'HIP08400.tsv', 'HIP08439.tsv', 'HIP08507.tsv',
       'HIP08521.tsv', 'HIP08596.tsv', 'HIP08598.tsv', 'HIP08653.tsv',
       'HIP08702.tsv', 'HIP08710.tsv', 'HIP08711.tsv', 'HIP08725.tsv',
       'HIP08792.tsv', 'HIP08805.tsv', 'HIP08816.tsv', 'HIP08821.tsv',
       'HIP08827.tsv', 'HIP08888.tsv', 'HIP08890.tsv', 'HIP08972.tsv',
       'HIP08977.tsv', 'HIP08986.tsv', 'HIP08989.tsv', 'HIP09001.tsv',
       'HIP09020.tsv', 'HIP09022.tsv', 'HIP09026.tsv', 'HIP09029.tsv',
       'HIP09041.tsv', 'HIP09046.tsv', 'HIP09051.tsv', 'HIP09062.tsv',
       'HIP09097.tsv', 'HIP09118.tsv', 'HIP09119.tsv', 'HIP09122.tsv',
       'HIP09150.tsv', 'HIP09159.tsv', 'HIP09190.tsv', 'HIP09235.tsv',
       'HIP09253.tsv', 'HIP09284.tsv', 'HIP09344.tsv', 'HIP09364.tsv',
       'HIP09365.tsv', 'HIP09366.tsv', 'HIP09430.tsv', 'HIP09559.tsv',
       'HIP09624.tsv', 'HIP09681.tsv', 'HIP09775.tsv', 'HIP09789.tsv',
       'HIP10358.tsv', 'HIP10376.tsv', 'HIP10377.tsv', 'HIP10389.tsv',
       'HIP10408.tsv', 'HIP10424.tsv', 'HIP10443.tsv', 'HIP10445.tsv',
       'HIP10447.tsv', 'HIP10480.tsv', 'HIP10514.tsv', 'HIP10545.tsv',
       'HIP10564.tsv', 'HIP10568.tsv', 'HIP10639.tsv', 'HIP10669.tsv',
       'HIP10694.tsv', 'HIP10716.tsv', 'HIP10726.tsv', 'HIP10730.tsv',
       'HIP10746.tsv', 'HIP10759.tsv', 'HIP10787.tsv', 'HIP10814.tsv',
       'HIP10815.tsv', 'HIP10817.tsv', 'HIP10820.tsv', 'HIP10821.tsv',
       'HIP10823.tsv', 'HIP10846.tsv', 'HIP11058.tsv', 'HIP11513.tsv',
       'HIP11518.tsv', 'HIP11553.tsv', 'HIP11613.tsv', 'HIP11649.tsv',
       'HIP11711.tsv', 'HIP11717.tsv', 'HIP11758.tsv', 'HIP11774.tsv',
       'HIP11784.tsv', 'HIP11845.tsv', 'HIP11857.tsv', 'HIP11937.tsv',
       'HIP11989.tsv', 'HIP12034.tsv', 'HIP12088.tsv', 'HIP12091.tsv',
       'HIP12097.tsv', 'HIP12099.tsv', 'HIP12123.tsv', 'HIP12129.tsv',
       'HIP12143.tsv', 'HIP12165.tsv', 'HIP12527.tsv', 'HIP12533.tsv',
       'HIP12534.tsv', 'HIP12538.tsv', 'HIP12703.tsv', 'HIP12743.tsv',
       'HIP12900.tsv', 'HIP12980.tsv', 'HIP13015.tsv', 'HIP13122.tsv',
       'HIP13142.tsv', 'HIP13157.tsv', 'HIP13168.tsv', 'HIP13176.tsv',
       'HIP13178.tsv', 'HIP13183.tsv', 'HIP13185.tsv', 'HIP13193.tsv',
       'HIP13198.tsv', 'HIP13206.tsv', 'HIP13209.tsv', 'HIP13214.tsv',
       'HIP13217.tsv', 'HIP13220.tsv', 'HIP13227.tsv', 'HIP13228.tsv',
       'HIP13230.tsv', 'HIP13233.tsv', 'HIP13244.tsv', 'HIP13251.tsv',
       'HIP13252.tsv', 'HIP13256.tsv', 'HIP13263.tsv', 'HIP13265.tsv',
       'HIP13274.tsv', 'HIP13276.tsv', 'HIP13284.tsv', 'HIP13291.tsv',
       'HIP13294.tsv', 'HIP13296.tsv', 'HIP13303.tsv', 'HIP13306.tsv',
       'HIP13309.tsv', 'HIP13311.tsv', 'HIP13318.tsv', 'HIP13319.tsv',
       'HIP13324.tsv', 'HIP13325.tsv', 'HIP13350.tsv', 'HIP13352.tsv',
       'HIP13355.tsv', 'HIP13360.tsv', 'HIP13361.tsv', 'HIP13363.tsv',
       'HIP13370.tsv', 'HIP13376.tsv', 'HIP13383.tsv', 'HIP13396.tsv',
       'HIP13402.tsv', 'HIP13414.tsv', 'HIP13427.tsv', 'HIP13449.tsv',
       'HIP13463.tsv', 'HIP13465.tsv', 'HIP13473.tsv', 'HIP13478.tsv',
       'HIP13489.tsv', 'HIP13497.tsv', 'HIP13505.tsv', 'HIP13511.tsv',
       'HIP13513.tsv', 'HIP13515.tsv', 'HIP13518.tsv', 'HIP13554.tsv',
       'HIP13567.tsv', 'HIP13592.tsv', 'HIP13610.tsv', 'HIP13625.tsv',
       'HIP13627.tsv', 'HIP13636.tsv', 'HIP13654.tsv', 'HIP13658.tsv',
       'HIP13661.tsv', 'HIP13667.tsv', 'HIP13671.tsv', 'HIP13686.tsv',
       'HIP13695.tsv', 'HIP13703.tsv', 'HIP13710.tsv', 'HIP13720.tsv',
       'HIP13722.tsv', 'HIP13746.tsv', 'HIP13749.tsv', 'HIP13757.tsv',
       'HIP13760.tsv', 'HIP13766.tsv', 'HIP13769.tsv', 'HIP13771.tsv',
       'HIP13773.tsv', 'HIP13777.tsv', 'HIP13780.tsv', 'HIP13782.tsv',
       'HIP13786.tsv', 'HIP13789.tsv', 'HIP13794.tsv', 'HIP13796.tsv',
       'HIP13800.tsv', 'HIP13803.tsv', 'HIP13806.tsv', 'HIP13809.tsv',
       'HIP13814.tsv', 'HIP13818.tsv', 'HIP13822.tsv', 'HIP13831.tsv',
       'HIP13833.tsv', 'HIP13847.tsv', 'HIP13852.tsv', 'HIP13853.tsv',
       'HIP13856.tsv', 'HIP13857.tsv', 'HIP13859.tsv', 'HIP13860.tsv',
       'HIP13865.tsv', 'HIP13869.tsv', 'HIP13871.tsv', 'HIP13875.tsv',
       'HIP13877.tsv', 'HIP13880.tsv', 'HIP13887.tsv', 'HIP13893.tsv',
       'HIP13894.tsv', 'HIP13900.tsv', 'HIP13902.tsv', 'HIP13903.tsv',
       'HIP13911.tsv', 'HIP13916.tsv', 'HIP13919.tsv', 'HIP13920.tsv',
       'HIP13923.tsv', 'HIP13926.tsv', 'HIP13928.tsv', 'HIP13929.tsv',
       'HIP13932.tsv', 'HIP13933.tsv', 'HIP13935.tsv', 'HIP13938.tsv',
       'HIP13939.tsv', 'HIP13941.tsv', 'HIP13944.tsv', 'HIP13945.tsv',
       'HIP13947.tsv', 'HIP13949.tsv', 'HIP13951.tsv', 'HIP13954.tsv',
       'HIP13956.tsv', 'HIP13958.tsv', 'HIP13961.tsv', 'HIP13962.tsv',
       'HIP13964.tsv', 'HIP13966.tsv', 'HIP13967.tsv', 'HIP13972.tsv',
       'HIP13975.tsv', 'HIP13976.tsv', 'HIP13978.tsv', 'HIP13981.tsv',
       'HIP13983.tsv', 'HIP13986.tsv', 'HIP13987.tsv', 'HIP13988.tsv',
       'HIP13989.tsv', 'HIP13992.tsv', 'HIP13994.tsv', 'HIP13996.tsv',
       'HIP14000.tsv', 'HIP14004.tsv', 'HIP14007.tsv', 'HIP14009.tsv',
       'HIP14014.tsv', 'HIP14015.tsv', 'HIP14016.tsv', 'HIP14018.tsv',
       'HIP14020.tsv', 'HIP14022.tsv', 'HIP14024.tsv', 'HIP14028.tsv',
       'HIP14030.tsv', 'HIP14034.tsv', 'HIP14036.tsv', 'HIP14037.tsv',
       'HIP14039.tsv', 'HIP14041.tsv', 'HIP14043.tsv', 'HIP14045.tsv',
       'HIP14048.tsv', 'HIP14051.tsv', 'HIP14053.tsv', 'HIP14055.tsv',
       'HIP14059.tsv', 'HIP14060.tsv', 'HIP14064.tsv', 'HIP14066.tsv',
       'HIP14071.tsv', 'HIP14072.tsv', 'HIP14074.tsv', 'HIP14077.tsv',
       'HIP14079.tsv', 'HIP14080.tsv', 'HIP14089.tsv', 'HIP14090.tsv',
       'HIP14092.tsv', 'HIP14095.tsv', 'HIP14096.tsv', 'HIP14103.tsv',
       'HIP14106.tsv', 'HIP14107.tsv', 'HIP14109.tsv', 'HIP14110.tsv',
       'HIP14114.tsv', 'HIP14118.tsv', 'HIP14121.tsv', 'HIP14124.tsv',
       'HIP14127.tsv', 'HIP14129.tsv', 'HIP14130.tsv', 'HIP14134.tsv',
       'HIP14136.tsv', 'HIP14138.tsv', 'HIP14140.tsv', 'HIP14142.tsv',
       'HIP14143.tsv', 'HIP14148.tsv', 'HIP14152.tsv', 'HIP14153.tsv',
       'HIP14156.tsv', 'HIP14157.tsv', 'HIP14160.tsv', 'HIP14161.tsv',
       'HIP14170.tsv', 'HIP14172.tsv', 'HIP14174.tsv', 'HIP14175.tsv',
       'HIP14176.tsv', 'HIP14178.tsv', 'HIP14181.tsv', 'HIP14183.tsv',
       'HIP14184.tsv', 'HIP14187.tsv', 'HIP14192.tsv', 'HIP14194.tsv',
       'HIP14196.tsv', 'HIP14202.tsv', 'HIP14205.tsv', 'HIP14206.tsv',
       'HIP14209.tsv', 'HIP14211.tsv', 'HIP14213.tsv', 'HIP14214.tsv',
       'HIP14217.tsv', 'HIP14218.tsv', 'HIP14221.tsv', 'HIP14223.tsv',
       'HIP14226.tsv', 'HIP14227.tsv', 'HIP14230.tsv', 'HIP14231.tsv',
       'HIP14234.tsv', 'HIP14236.tsv', 'HIP14237.tsv', 'HIP14238.tsv',
       'HIP14240.tsv', 'HIP14241.tsv', 'HIP14243.tsv', 'HIP14244.tsv',
       'HIP14361.tsv', 'HIP14363.tsv', 'HIP14494.tsv', 'HIP14844.tsv',
       'HIP14911.tsv', 'HIP15685.tsv', 'HIP15854.tsv', 'HIP15855.tsv',
       'HIP15860.tsv', 'HIP15861.tsv', 'HIP16515.tsv', 'HIP16738.tsv',
       'HIP16867.tsv', 'HIP17370.tsv', 'HIP17445.tsv', 'HIP17449.tsv',
       'HIP17457.tsv', 'HIP17462.tsv', 'HIP17534.tsv', 'HIP17577.tsv',
       'HIP17585.tsv', 'HIP17657.tsv', 'HIP17698.tsv', 'HIP17723.tsv',
       'HIP17737.tsv', 'HIP17760.tsv', 'HIP17793.tsv', 'HIP17837.tsv',
       'HIP17845.tsv', 'HIP17887.tsv', 'HIP19048.tsv', 'HIP19089.tsv',
       'HIP19716.tsv', 'HIP19717.tsv'])

PATH = sys.argv[1]

print(f'PATH={PATH}')
def sign_to_int(sign):
    if sign == '-' : return 0
    if sign == '+' : return 1 
    
def get_tag(tag, name):
    l = [x.split(name)[1] for x in tag if x.startswith(name)]
    #print(l)
    if len(l) == 0 : return 'Unknown'
    if len(l) == 1 : return l[0]
    else : return l
    

    
def parse_batch1(filename, top_n):
    """Reads .tsvs from emerson data and parse the sample tags & filters"""
    #Basic reading and filtering
    tmp = pd.read_csv(filename, sep='\t', usecols = ['amino_acid', 'frequency','v_family','v_gene',
                                                     'j_family','j_gene','d_family','d_gene', 'sample_tags']).dropna()
    tmp['filename']= os.path.basename(filename)
    #Filtering NaNs, unresolved, bad sequences
    tmp.dropna(subset=['amino_acid','v_family',	'v_gene','j_family','j_gene'])
    tmp = tmp.query('not (amino_acid.str.contains("\*") or v_family == "TCRBVA")')
    tmp = tmp.query('v_gene!="unresolved" & j_gene != "unresolved"').dropna()
    tmp = tmp.sort_values('frequency', ascending=False).head(top_n)

    #Parsing tags
    tags = tmp.sample_tags.unique()[0].split(',')
    age = get_tag(tags, 'Age:')
    if get_tag(tags, 'Virus Diseases:') == 'Unknown':
        tmp['pred_cmv'] = 'Unknown'
        tmp['true_cmv'] = 'Unknown'
    else: 
        tmp['pred_cmv'] = sign_to_int(get_tag(tags, 'Inferred CMV status:').split('CMV ')[1])
        tmp['true_cmv'] = sign_to_int(get_tag(tags, 'Virus Diseases:').split('Cytomegalovirus ')[1])
    
    #Bypassing blank age
    if age == 'Unknown': tmp['age'] = age
    else: tmp['age'] = int(age.split(' ')[0])
    #no issue here
    tmp['sex'] = get_tag(tags,'Biological Sex:')
    
    #getting unknown instead of unknown racial group
    race = get_tag(tags, 'Racial').split(':')[1]
    if race.startswith('Unknown'):
        tmp['race'] = 'Unknown'
    else : tmp['race'] = race
        
    #Initializing columns to Unknown to facilitate labels parsing in case of missing alleles
    tmp[['hla_a1', 'hla_a2', 'hla_b1', 'hla_b2']]= 'Unknown'
    
    #Getting HLA columns, then the A and B labels
    HLA = get_tag(tags,'HLA MHC class I:')
    a = len([x for x in HLA if x.startswith('HLA-A')])
    b = len([x for x in HLA if x.startswith('HLA-B')])
    for x in range(a):
        tmp['hla_a'+str(x+1)] = ''.join(HLA.pop(0).split('HLA-')[1].split('*'))
    for x in range(b):
        tmp['hla_b'+str(x+1)] = ''.join(HLA.pop(0).split('HLA-')[1].split('*'))
    

    tmp.drop(columns='sample_tags',inplace=True)
    return tmp

#####################################################################333
# Parsing batch1
zfile = 'emerson-2017-natgen.zip'

emb1 = pd.DataFrame(columns=['amino_acid', 'frequency', 'v_family', 'v_gene', 'd_family', 'd_gene', 'j_family', 'j_gene', 'filename', 'pred_cmv', 'true_cmv', 'age', 'sex',
       'race', 'hla_a1', 'hla_a2', 'hla_b1', 'hla_b2'])


from tqdm import tqdm 
#####################################################################333

print('unzipping & parsing tag/sequences')
with zf.ZipFile(PATH+zfile, 'r') as zipped:
    #Can't parse all at the same time, so I unzip, then parse each file then delete it
    BATCH1 = sorted(names,key=str) #HERE PARSING ALL the files that have at least 1 known HLA
    for file in tqdm(BATCH1, desc='File / 626 :'):
        zipped.extract(file,PATH)
        filename = os.path.join(PATH,file)
        emb1=emb1.append(parse_batch1(filename,40000), ignore_index=True)
        os.remove(filename)

print('filtering abherrant sequences')       
not_C = emb1.query('not amino_acid.str.startswith("C")').index
not_F = emb1.query('not amino_acid.str.endswith("F")').index
both = not_C.union(not_F)
emb1.drop(index=both, inplace=True)
emb1.drop(index=emb1.query('len<=9').index, inplace=True)

#####################################################################
print('getting relative frequency (relative to top 40k)')
rel_frq = np.empty(0,dtype=np.float64)
for f in emb1.filename.unique():
    x = emb1.query('filename==@f')['frequency'].values
    rel_frq=np.append(rel_frq, x/max(x))
emb1['rel_freq']=rel_frq

emb1 = emb1[['amino_acid', 'frequency', 'rel_freq', 'v_family', 'v_gene', 'd_family', 'd_gene',
       'j_family', 'j_gene', 'filename', 'pred_cmv', 'true_cmv', 'age', 'sex',
       'race', 'hla_a1', 'hla_a2', 'hla_b1', 'hla_b2', 'len']]



#####################################################################333
print('splitting into train/test based on number of resolved HLA')
#Splitting into train/test based on number of resolved HLA

def sample_topn(df, n):
    """Group by filename, sort by TCR frequency, and take top n sequences"""
    grp = df.sort_values('frequency', ascending = False).groupby('filename')
    #Here I take the minimum between 
    sample_map = {k:min(n,v) for (k,v) in zip(sorted(df.filename.unique(), key=str),
                                              grp.size().sort_index().values)}
    return grp.apply(lambda group: group.head(sample_map[group.name])).reset_index(drop=True)
print('reading tag file')

files_test = ['HIP00734.tsv', 'HIP00822.tsv', 'HIP00825.tsv', 'HIP00926.tsv',
        'HIP01004.tsv', 'HIP01384.tsv', 'HIP01501.tsv', 'HIP01571.tsv',
        'HIP02103.tsv', 'HIP02873.tsv', 'HIP03236.tsv', 'HIP03651.tsv',
        'HIP03678.tsv', 'HIP04545.tsv', 'HIP04576.tsv', 'HIP05311.tsv',
        'HIP05388.tsv', 'HIP05409.tsv', 'HIP05437.tsv', 'HIP05444.tsv',
        'HIP05467.tsv', 'HIP08223.tsv', 'HIP09062.tsv', 'HIP09159.tsv',
        'HIP09364.tsv', 'HIP10358.tsv', 'HIP10445.tsv', 'HIP10480.tsv',
        'HIP10514.tsv', 'HIP10564.tsv', 'HIP10669.tsv', 'HIP10787.tsv',
        'HIP10817.tsv', 'HIP10846.tsv', 'HIP11649.tsv', 'HIP11711.tsv',
        'HIP12097.tsv', 'HIP12900.tsv', 'HIP13284.tsv', 'HIP13324.tsv',
        'HIP13402.tsv', 'HIP13449.tsv', 'HIP13661.tsv', 'HIP13794.tsv',
        'HIP13865.tsv', 'HIP13903.tsv', 'HIP13941.tsv', 'HIP13961.tsv',
        'HIP14004.tsv', 'HIP14028.tsv', 'HIP14060.tsv', 'HIP14071.tsv',
        'HIP14072.tsv', 'HIP14079.tsv', 'HIP14080.tsv', 'HIP14089.tsv',
        'HIP14227.tsv', 'HIP14361.tsv', 'HIP15854.tsv', 'HIP17657.tsv',
        'HIP17737.tsv', 'HIP19717.tsv']

files_train = ['HIP00110.tsv', 'HIP00169.tsv', 'HIP00594.tsv', 'HIP00602.tsv',
        'HIP00614.tsv', 'HIP00640.tsv', 'HIP00707.tsv', 'HIP00710.tsv',
        'HIP00715.tsv', 'HIP00728.tsv', 'HIP00761.tsv', 'HIP00769.tsv',
        'HIP00771.tsv', 'HIP00773.tsv', 'HIP00775.tsv', 'HIP00777.tsv',
        'HIP00779.tsv', 'HIP00805.tsv', 'HIP00813.tsv', 'HIP00819.tsv',
        'HIP00826.tsv', 'HIP00832.tsv', 'HIP00838.tsv', 'HIP00851.tsv',
        'HIP00869.tsv', 'HIP00898.tsv', 'HIP00904.tsv', 'HIP00924.tsv',
        'HIP00934.tsv', 'HIP00951.tsv', 'HIP00985.tsv', 'HIP00997.tsv',
        'HIP00999.tsv', 'HIP01022.tsv', 'HIP01055.tsv', 'HIP01129.tsv',
        'HIP01140.tsv', 'HIP01160.tsv', 'HIP01162.tsv', 'HIP01180.tsv',
        'HIP01181.tsv', 'HIP01197.tsv', 'HIP01206.tsv', 'HIP01218.tsv',
        'HIP01220.tsv', 'HIP01223.tsv', 'HIP01232.tsv', 'HIP01255.tsv',
        'HIP01264.tsv', 'HIP01266.tsv', 'HIP01298.tsv', 'HIP01313.tsv',
        'HIP01359.tsv', 'HIP01391.tsv', 'HIP01392.tsv', 'HIP01393.tsv',
        'HIP01465.tsv', 'HIP01470.tsv', 'HIP01499.tsv', 'HIP01582.tsv',
        'HIP01795.tsv', 'HIP01797.tsv', 'HIP01798.tsv', 'HIP01805.tsv',
        'HIP01820.tsv', 'HIP01850.tsv', 'HIP01856.tsv', 'HIP01865.tsv',
        'HIP01867.tsv', 'HIP01870.tsv', 'HIP01947.tsv', 'HIP02024.tsv',
        'HIP02078.tsv', 'HIP02090.tsv', 'HIP02112.tsv', 'HIP02126.tsv',
        'HIP02663.tsv', 'HIP02734.tsv', 'HIP02737.tsv', 'HIP02742.tsv',
        'HIP02805.tsv', 'HIP02811.tsv', 'HIP02820.tsv', 'HIP02848.tsv',
        'HIP02855.tsv', 'HIP02875.tsv', 'HIP02877.tsv', 'HIP02928.tsv',
        'HIP02931.tsv', 'HIP02947.tsv', 'HIP02962.tsv', 'HIP02997.tsv',
        'HIP03004.tsv', 'HIP03099.tsv', 'HIP03107.tsv', 'HIP03111.tsv',
        'HIP03184.tsv', 'HIP03228.tsv', 'HIP03233.tsv', 'HIP03275.tsv',
        'HIP03370.tsv', 'HIP03378.tsv', 'HIP03381.tsv', 'HIP03383.tsv',
        'HIP03385.tsv', 'HIP03484.tsv', 'HIP03494.tsv', 'HIP03495.tsv',
        'HIP03502.tsv', 'HIP03505.tsv', 'HIP03511.tsv', 'HIP03591.tsv',
        'HIP03592.tsv', 'HIP03597.tsv', 'HIP03618.tsv', 'HIP03628.tsv',
        'HIP03630.tsv', 'HIP03677.tsv', 'HIP03685.tsv', 'HIP03693.tsv',
        'HIP03695.tsv', 'HIP03720.tsv', 'HIP03732.tsv', 'HIP03807.tsv',
        'HIP03812.tsv', 'HIP04455.tsv', 'HIP04464.tsv', 'HIP04471.tsv',
        'HIP04475.tsv', 'HIP04480.tsv', 'HIP04498.tsv', 'HIP04509.tsv',
        'HIP04510.tsv', 'HIP04511.tsv', 'HIP04527.tsv', 'HIP04532.tsv',
        'HIP04552.tsv', 'HIP04555.tsv', 'HIP04578.tsv', 'HIP04597.tsv',
        'HIP04605.tsv', 'HIP04611.tsv', 'HIP04634.tsv', 'HIP04958.tsv',
        'HIP05331.tsv', 'HIP05377.tsv', 'HIP05390.tsv', 'HIP05398.tsv',
        'HIP05405.tsv', 'HIP05434.tsv', 'HIP05455.tsv', 'HIP05460.tsv',
        'HIP05524.tsv', 'HIP05533.tsv', 'HIP05535.tsv', 'HIP05540.tsv',
        'HIP05551.tsv', 'HIP05552.tsv', 'HIP05559.tsv', 'HIP05561.tsv',
        'HIP05563.tsv', 'HIP05574.tsv', 'HIP05578.tsv', 'HIP05590.tsv',
        'HIP05595.tsv', 'HIP05665.tsv', 'HIP05757.tsv', 'HIP05763.tsv',
        'HIP05815.tsv', 'HIP05817.tsv', 'HIP05832.tsv', 'HIP05838.tsv',
        'HIP05841.tsv', 'HIP05934.tsv', 'HIP05941.tsv', 'HIP05942.tsv',
        'HIP05948.tsv', 'HIP05960.tsv', 'HIP06191.tsv', 'HIP07754.tsv',
        'HIP08076.tsv', 'HIP08200.tsv', 'HIP08230.tsv', 'HIP08236.tsv',
        'HIP08305.tsv', 'HIP08337.tsv', 'HIP08339.tsv', 'HIP08345.tsv',
        'HIP08346.tsv', 'HIP08389.tsv', 'HIP08399.tsv', 'HIP08400.tsv',
        'HIP08439.tsv', 'HIP08507.tsv', 'HIP08521.tsv', 'HIP08596.tsv',
        'HIP08598.tsv', 'HIP08653.tsv', 'HIP08702.tsv', 'HIP08710.tsv',
        'HIP08711.tsv', 'HIP08725.tsv', 'HIP08792.tsv', 'HIP08805.tsv',
        'HIP08816.tsv', 'HIP08821.tsv', 'HIP08827.tsv', 'HIP08888.tsv',
        'HIP08890.tsv', 'HIP08972.tsv', 'HIP08977.tsv', 'HIP08986.tsv',
        'HIP08989.tsv', 'HIP09001.tsv', 'HIP09020.tsv', 'HIP09022.tsv',
        'HIP09026.tsv', 'HIP09029.tsv', 'HIP09041.tsv', 'HIP09046.tsv',
        'HIP09051.tsv', 'HIP09097.tsv', 'HIP09118.tsv', 'HIP09119.tsv',
        'HIP09122.tsv', 'HIP09150.tsv', 'HIP09190.tsv', 'HIP09235.tsv',
        'HIP09253.tsv', 'HIP09284.tsv', 'HIP09344.tsv', 'HIP09365.tsv',
        'HIP09366.tsv', 'HIP09430.tsv', 'HIP09559.tsv', 'HIP09624.tsv',
        'HIP09681.tsv', 'HIP09775.tsv', 'HIP09789.tsv', 'HIP10376.tsv',
        'HIP10377.tsv', 'HIP10389.tsv', 'HIP10408.tsv', 'HIP10424.tsv',
        'HIP10443.tsv', 'HIP10447.tsv', 'HIP10545.tsv', 'HIP10568.tsv',
        'HIP10639.tsv', 'HIP10694.tsv', 'HIP10716.tsv', 'HIP10726.tsv',
        'HIP10730.tsv', 'HIP10746.tsv', 'HIP10759.tsv', 'HIP10814.tsv',
        'HIP10815.tsv', 'HIP10820.tsv', 'HIP10821.tsv', 'HIP10823.tsv',
        'HIP11058.tsv', 'HIP11513.tsv', 'HIP11518.tsv', 'HIP11553.tsv',
        'HIP11613.tsv', 'HIP11717.tsv', 'HIP11758.tsv', 'HIP11774.tsv',
        'HIP11784.tsv', 'HIP11845.tsv', 'HIP11857.tsv', 'HIP11937.tsv',
        'HIP11989.tsv', 'HIP12034.tsv', 'HIP12088.tsv', 'HIP12091.tsv',
        'HIP12099.tsv', 'HIP12123.tsv', 'HIP12129.tsv', 'HIP12143.tsv',
        'HIP12165.tsv', 'HIP12527.tsv', 'HIP12533.tsv', 'HIP12534.tsv',
        'HIP12538.tsv', 'HIP12703.tsv', 'HIP12743.tsv', 'HIP12980.tsv',
        'HIP13015.tsv', 'HIP13122.tsv', 'HIP13142.tsv', 'HIP13157.tsv',
        'HIP13168.tsv', 'HIP13176.tsv', 'HIP13178.tsv', 'HIP13183.tsv',
        'HIP13185.tsv', 'HIP13193.tsv', 'HIP13198.tsv', 'HIP13206.tsv',
        'HIP13209.tsv', 'HIP13214.tsv', 'HIP13217.tsv', 'HIP13220.tsv',
        'HIP13227.tsv', 'HIP13228.tsv', 'HIP13230.tsv', 'HIP13233.tsv',
        'HIP13244.tsv', 'HIP13251.tsv', 'HIP13252.tsv', 'HIP13256.tsv',
        'HIP13263.tsv', 'HIP13265.tsv', 'HIP13274.tsv', 'HIP13276.tsv',
        'HIP13291.tsv', 'HIP13294.tsv', 'HIP13296.tsv', 'HIP13303.tsv',
        'HIP13306.tsv', 'HIP13309.tsv', 'HIP13311.tsv', 'HIP13318.tsv',
        'HIP13319.tsv', 'HIP13325.tsv', 'HIP13350.tsv', 'HIP13352.tsv',
        'HIP13355.tsv', 'HIP13360.tsv', 'HIP13361.tsv', 'HIP13363.tsv',
        'HIP13370.tsv', 'HIP13376.tsv', 'HIP13383.tsv', 'HIP13396.tsv',
        'HIP13414.tsv', 'HIP13427.tsv', 'HIP13463.tsv', 'HIP13465.tsv',
        'HIP13473.tsv', 'HIP13478.tsv', 'HIP13489.tsv', 'HIP13497.tsv',
        'HIP13505.tsv', 'HIP13511.tsv', 'HIP13513.tsv', 'HIP13515.tsv',
        'HIP13518.tsv', 'HIP13554.tsv', 'HIP13567.tsv', 'HIP13592.tsv',
        'HIP13610.tsv', 'HIP13625.tsv', 'HIP13627.tsv', 'HIP13636.tsv',
        'HIP13654.tsv', 'HIP13658.tsv', 'HIP13667.tsv', 'HIP13671.tsv',
        'HIP13686.tsv', 'HIP13695.tsv', 'HIP13703.tsv', 'HIP13710.tsv',
        'HIP13720.tsv', 'HIP13722.tsv', 'HIP13746.tsv', 'HIP13749.tsv',
        'HIP13757.tsv', 'HIP13760.tsv', 'HIP13766.tsv', 'HIP13769.tsv',
        'HIP13771.tsv', 'HIP13773.tsv', 'HIP13777.tsv', 'HIP13780.tsv',
        'HIP13782.tsv', 'HIP13786.tsv', 'HIP13789.tsv', 'HIP13796.tsv',
        'HIP13800.tsv', 'HIP13803.tsv', 'HIP13806.tsv', 'HIP13809.tsv',
        'HIP13814.tsv', 'HIP13818.tsv', 'HIP13822.tsv', 'HIP13831.tsv',
        'HIP13833.tsv', 'HIP13847.tsv', 'HIP13852.tsv', 'HIP13853.tsv',
        'HIP13856.tsv', 'HIP13857.tsv', 'HIP13859.tsv', 'HIP13860.tsv',
        'HIP13869.tsv', 'HIP13871.tsv', 'HIP13875.tsv', 'HIP13877.tsv',
        'HIP13880.tsv', 'HIP13887.tsv', 'HIP13893.tsv', 'HIP13894.tsv',
        'HIP13900.tsv', 'HIP13902.tsv', 'HIP13911.tsv', 'HIP13916.tsv',
        'HIP13919.tsv', 'HIP13920.tsv', 'HIP13923.tsv', 'HIP13926.tsv',
        'HIP13928.tsv', 'HIP13929.tsv', 'HIP13932.tsv', 'HIP13933.tsv',
        'HIP13935.tsv', 'HIP13938.tsv', 'HIP13939.tsv', 'HIP13944.tsv',
        'HIP13945.tsv', 'HIP13947.tsv', 'HIP13949.tsv', 'HIP13951.tsv',
        'HIP13954.tsv', 'HIP13956.tsv', 'HIP13958.tsv', 'HIP13962.tsv',
        'HIP13964.tsv', 'HIP13966.tsv', 'HIP13967.tsv', 'HIP13972.tsv',
        'HIP13975.tsv', 'HIP13976.tsv', 'HIP13978.tsv', 'HIP13981.tsv',
        'HIP13983.tsv', 'HIP13986.tsv', 'HIP13987.tsv', 'HIP13988.tsv',
        'HIP13989.tsv', 'HIP13992.tsv', 'HIP13994.tsv', 'HIP13996.tsv',
        'HIP14000.tsv', 'HIP14007.tsv', 'HIP14009.tsv', 'HIP14014.tsv',
        'HIP14015.tsv', 'HIP14016.tsv', 'HIP14018.tsv', 'HIP14020.tsv',
        'HIP14022.tsv', 'HIP14024.tsv', 'HIP14030.tsv', 'HIP14034.tsv',
        'HIP14036.tsv', 'HIP14037.tsv', 'HIP14039.tsv', 'HIP14041.tsv',
        'HIP14043.tsv', 'HIP14045.tsv', 'HIP14048.tsv', 'HIP14051.tsv',
        'HIP14053.tsv', 'HIP14055.tsv', 'HIP14059.tsv', 'HIP14064.tsv',
        'HIP14066.tsv', 'HIP14074.tsv', 'HIP14077.tsv', 'HIP14090.tsv',
        'HIP14092.tsv', 'HIP14095.tsv', 'HIP14096.tsv', 'HIP14103.tsv',
        'HIP14106.tsv', 'HIP14107.tsv', 'HIP14109.tsv', 'HIP14110.tsv',
        'HIP14114.tsv', 'HIP14118.tsv', 'HIP14121.tsv', 'HIP14124.tsv',
        'HIP14127.tsv', 'HIP14129.tsv', 'HIP14130.tsv', 'HIP14134.tsv',
        'HIP14136.tsv', 'HIP14138.tsv', 'HIP14140.tsv', 'HIP14142.tsv',
        'HIP14143.tsv', 'HIP14148.tsv', 'HIP14152.tsv', 'HIP14153.tsv',
        'HIP14156.tsv', 'HIP14157.tsv', 'HIP14160.tsv', 'HIP14161.tsv',
        'HIP14170.tsv', 'HIP14172.tsv', 'HIP14174.tsv', 'HIP14175.tsv',
        'HIP14176.tsv', 'HIP14178.tsv', 'HIP14181.tsv', 'HIP14183.tsv',
        'HIP14184.tsv', 'HIP14187.tsv', 'HIP14192.tsv', 'HIP14194.tsv',
        'HIP14196.tsv', 'HIP14202.tsv', 'HIP14205.tsv', 'HIP14206.tsv',
        'HIP14209.tsv', 'HIP14211.tsv', 'HIP14213.tsv', 'HIP14214.tsv',
        'HIP14217.tsv', 'HIP14218.tsv', 'HIP14221.tsv', 'HIP14223.tsv',
        'HIP14226.tsv', 'HIP14230.tsv', 'HIP14231.tsv', 'HIP14234.tsv',
        'HIP14236.tsv', 'HIP14237.tsv', 'HIP14238.tsv', 'HIP14240.tsv',
        'HIP14241.tsv', 'HIP14243.tsv', 'HIP14244.tsv', 'HIP14363.tsv',
        'HIP14494.tsv', 'HIP14844.tsv', 'HIP14911.tsv', 'HIP15685.tsv',
        'HIP15855.tsv', 'HIP15860.tsv', 'HIP15861.tsv', 'HIP16515.tsv',
        'HIP16738.tsv', 'HIP16867.tsv', 'HIP17370.tsv', 'HIP17445.tsv',
        'HIP17449.tsv', 'HIP17457.tsv', 'HIP17462.tsv', 'HIP17534.tsv',
        'HIP17577.tsv', 'HIP17585.tsv', 'HIP17698.tsv', 'HIP17723.tsv',
        'HIP17760.tsv', 'HIP17793.tsv', 'HIP17837.tsv', 'HIP17845.tsv',
        'HIP17887.tsv', 'HIP19048.tsv', 'HIP19089.tsv', 'HIP19716.tsv']

train = emb1.query('filename in @files_train')
test = emb1.query('filename in @files_test')
print('saving to train/test')
train.to_csv(os.path.join(PATH,'emerson_batch1_train.tsv'), sep='\t',
             header = True, index = False)
test.to_csv(os.path.join(PATH,'emerson_batch1_test.tsv'), sep='\t',
            header = True, index = False)

print('sub sampling to top 10k sequences')
sub_test = sample_topn(test, 10000)
sub_train = sample_topn(train, 10000)

sub_train.to_csv(PATH+'emerson_batch1_train_top10k_hla.tsv',
           sep='\t', header= True, index=False)
sub_test.to_csv(PATH+'emerson_batch1_test_top10k_hla.tsv',
           sep='\t', header= True, index=False)