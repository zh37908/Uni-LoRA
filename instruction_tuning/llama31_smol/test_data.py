from common import *
def main():
    tok=load_tokenizer()
    messages=[{'role':'system','content':'Be precise.'},{'role':'user','content':'计算 2 + 2。'},{'role':'assistant','content':'4'},{'role':'user','content':'And 3 + 4?'},{'role':'assistant','content':'7'}]
    row=encode_chat(messages,tok,2048)
    assert row and row['labels'].count(128001)==2
    assert tok.decode([v for v in row['labels'] if v!=-100])=='4<|end_of_text|>7<|end_of_text|>'
    prefix,_=serialize(messages[:-1],tok,True)
    assert row['input_ids'][:len(prefix)]==prefix and all(x==-100 for x in row['labels'][len(prefix)-len(tok.encode(header('assistant'),add_special_tokens=False)):len(prefix)])
    assert encode_chat(messages,tok,row['length']) is None
    assert encode_chat(messages[:-1],tok,2048) is None
    assert encode_chat([{'role':'user','content':'x'},{'role':'assistant','content':'<|end_of_text|>'}],tok,2048) is None
    shorter=encode_chat(messages[:3],tok,2048)
    b=collate([row,shorter],128001,'cpu')
    assert int(b['attention_mask'][1].sum())==shorter['length']
    assert b['labels'][1,shorter['length']-1]==128001
    assert (b['labels'][1,shorter['length']:]==-100).all()
    print('DATA_TESTS_PASSED',flush=True)
if __name__=='__main__':main()
