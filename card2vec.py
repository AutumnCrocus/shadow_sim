# -*- coding: utf-8 -*-
import json
#import MeCab
from gensim.models import doc2vec
import os


def load_json(target_game_name):
    # カード名とカードテキストの入力データ作成
    names = []
    text = ""
    texts = []

    # Mecabの出力を分かち書きに指定
    #mecab = MeCab.Tagger("-Owakati")

    json_path = target_game_name + ".json"
    #import stanza
    #nlp = stanza.Pipeline(lang='en', processors='tokenize,lemma')
    # カードのテキストを形態素解析し、分かち書きしたものを改行区切りで一つのstringにする
    with open(json_path, "r") as file:
        card_dict = json.load(file)
        card_name_list = []
        for card_id in card_dict:
           card_name_list.append(card_dict[card_id]["name_"])
        for card_id in card_dict:
            card = card_dict[card_id]
            #print(card.keys())
            if card["name_"] not in names:
                names.append(card["name_"])
                pp = str(card["pp_"]) #"pp: " +
                card_type = card["type_"] #"type: " +
                """
                pp = "pp: " + str(card["pp_"])
                craft = "craft: " + card["craft_"].replace("craft","")
                card_type = "type: " + card["type_"]
                base_atk = "baseAtk: -"
                base_def = "baseDef: -"
                evo_atk = "evoAtk : -"
                evo_def = "evoDef: -"
                if "Follower" in card_type:
                    base_atk = "baseAtk: " + str(card["baseAtk_"])
                    base_def = "baseDef: " + str(card["baseDef_"])
                    evo_atk = "evoAtk: " + str(card["evoAtk_"])
                    evo_def = "evoDef: " + str(card["evoDef_"])
                """
                base_effect = card["baseEffect_"] #"baseEffect: " +
                base_effect = base_effect.replace("\n","$")
                evo_effect = card["evoEffect_"] # "evoEffect: " +
                evo_effect = evo_effect.replace("\n","$")
                if "Same as the unevolved form" in evo_effect:
                    evo_effect = card["baseEffect_"]
                    if "----------" in evo_effect:
                        evo_effect = evo_effect.split("----------")[1]
                    except_words = ["Enhance","Accelerate","Crystallize","Fanfare","Invocation",\
                                    "When this follower comes into play"]
                    for except_word in except_words:
                        if except_word in evo_effect:
                            evo_effect = evo_effect.split("$")
                            evo_effect = [sentence for sentence in evo_effect if except_word not in sentence]
                            if len(evo_effect) > 0:
                                evo_effect = "$".join(evo_effect)
                            else:
                                evo_effect = "-"

                    default_text = card["baseEffect_"]
                    if "Fanfare" in default_text and "Last Words" in default_text:
                        evo_effect += "Last Words:" + default_text.split("Last Words:")[-1]
                    evo_effect = "evoEffect: " + evo_effect
                base_effect = base_effect.replace("$", " ")
                evo_effect = evo_effect.replace("$"," ")
                parse_result = base_effect + " " + evo_effect
                parse_result = parse_result.replace(".", " . ")
                parse_result = parse_result.replace("/", " / ")
                parse_result = parse_result.replace(":", " : ")
                parse_result = parse_result.lower()
                for name in card_name_list:
                    if name.lower() in parse_result:
                        new_name  = name.replace(" ","")
                        new_name = new_name.replace(", ",",")
                        parse_result = parse_result.replace(name.lower(),new_name)

                parse_result = parse_result.replace(", ", " , ")
                parse_result = pp + "\n" + card_type + "\n" + parse_result
                #doc = nlp(parse_result)
                #parse_result = " ".join([word.lemma for sent in doc.sentences for word in sent.words])

                """
                parse_result = craft + " " + card_type + "\n"
                parse_result += pp + "\n"
                parse_result += base_atk + " " + base_def + "\n"
                parse_result += base_effect + "\n"
                parse_result += evo_atk + " " + evo_def + "\n"
                parse_result += evo_effect
                """
                #mecab_result = mecab.parse(card["text"])
                if parse_result is False:
                    text += "\n"
                    texts.append("")
                else:
                    #text += parse_result + "\n"
                    texts.append(parse_result)
                    text += parse_result.replace("\n"," ") + "\n"

                #names.append(card["name_"]+"(EVO)")
                #parse_result=card["evoEffect_"]
                #parse_result = parse_result.replace("\n"," ")
                #if parse_result is False:
                #    text += "\n"
                #    texts.append("")
                #else:
                #    text += parse_result + "\n"
                #    texts.append(card["evoEffect_"])


    with open(target_game_name + ".txt", "w") as file:
        file.write(text)

    return names, texts


def generate_doc2vec_model(target_game_name,size=300,window=8):
    print("Training Start")
    # カードテキスト読み込み
    card_text = doc2vec.TaggedLineDocument(target_game_name + ".txt")
    # 学習
    model = doc2vec.Doc2Vec(card_text, size=size, window=window, min_count=1,
            workers=4, iter=400, dbow_words=1, negative=10,dm=0)

    # モデルの保存
    model.save(target_game_name + ".model")
    print("Training Finish")
    return model


if __name__ == '__main__':
    TARGET_GAME_NAME = "all"
    names, texts = load_json(TARGET_GAME_NAME)
    import argparse
    parser = argparse.ArgumentParser(description='Card2Vec学習コード')
    parser.add_argument('--vector_size', help='カードベクトルのサイズ', type=int, default=300)
    parser.add_argument('--window_size', help='窓サイズ', type=int, default=8)
    parser.add_argument('-visualize',help='ベクトルの可視化')
    args = parser.parse_args()
    if os.path.isfile(TARGET_GAME_NAME + ".model") is True:
        model = doc2vec.Doc2Vec.load(TARGET_GAME_NAME + ".model")
    else:
        model = generate_doc2vec_model(TARGET_GAME_NAME,size=args.vector_size,window=args.window_size)

    if args.visualize is not None:
        init_word = input("input initial word:")
        positive_word = input("input positive word:")
        negatice_word = ""#("input negative word:")
        query = model.most_similar(positive=[init_word,positive_word])#,negative=[negatice_word])
        print(query)
        #print(names[card_index])
        #print(texts[card_index])
        #print(model.docvecs[card_index])
    TARGET_CARD_NAME = input("input card_name:")
    card_index = names.index(TARGET_CARD_NAME)
    print(model.docvecs[card_index])
    print(model.docvecs["Goblin"])
    reverse_flg = input("reverse?(y/n):") == "y" 
    # 類似カードと類似度のタプル（類似度上位10件）のリストを受け取る
    similar_docs = model.docvecs.most_similar(card_index,topn=len(names))
    print("name_len:",len(names),"txt_len:",len(texts))
    print(names[card_index])
    print(texts[card_index])
    print("--------------------is similar to--------------------")
    if reverse_flg:
        similar_docs = list(reversed(similar_docs))
    for similar_doc in similar_docs[0:10]:
        #print("similar_doc:",similar_doc)
        print(names[similar_doc[0]] + " " + str(similar_doc[1]))
        print(texts[similar_doc[0]], "\n")
