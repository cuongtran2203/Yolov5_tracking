import numpy as np
import time
def fillter_output(outputs):
    # outputs=outputs.cpu().detach().numpy()
    out_list=[]
    out_cls=[]
    # print("outputs list: ", outputs)

    # dict_class={"0":[],"1":[],"2":[],"3":[],"4":[]}
    dict_class={}
    for output in outputs:
        cls=output[-1]
        dict_class.update({str(cls):[]})
    for output in outputs:
        cls=int(output[-1])
        if str(cls) in dict_class.keys():
            dict_class[str(cls)].append(output)
    return dict_class
if __name__ == "__main__":
    outputs=np.array([[1,155,1],[145,14,2],[146,258,1],[147,96,1],[85,1477,2],[147,69,4]])
    outputs=outputs.tolist()
    print("Org :",outputs)
    print("length :",len(outputs))
    print("Result :",fillter_output(outputs))