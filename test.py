records = [line.split("\t") for line in open(r"C:\Users\liuyuan\Desktop\sentiment\dataset\SentiWordNet.txt", "r")]
refine_scores = [records[i] for i, x in enumerate(records) if '#' not in x[0]]
print(refine_scores[0])




    
     
        

            



