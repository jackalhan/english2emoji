import numpy as np
import tensorflow as tf

batchBuyuklugu = 128

def sozlukOku(sozlukAdi):
    sozluk = {}
    dosya = open(sozlukAdi, encoding="utf-8")
    for kayit in dosya:
        sozluk[kayit[:-1]] = len(sozluk)
    return sozluk

def tokenlaraAyir(cumle, sozluk):
    cumle = cumle.replace(',', ' ,').replace('.', ' .').replace('\n', ' ')
    tokenlar = cumle.split(' ')
    tokY = []
    for tok in tokenlar:
        if tok not in sozluk.keys():
            tokY.append("<UNK>")
        else:
            tokY.append(tok)
    return tokY

kaynakSozlugu = sozlukOku("vocab.50K.en.txt")
hedefSozlugu = sozlukOku("vocab.50K.de.txt")

cumle = 'he saw a old yellow truck .'
cumle = tokenlaraAyir(cumle, kaynakSozlugu)

loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as session:
    loader = tf.train.import_meta_graph("\kayit" + '.meta')
    loader.restore(session, load_path)

    input_data = loaded_graph.get_tensor_by_name('input:0')
    logitler = loaded_graph.get_tensor_by_name('predictions:0')
    hedefSeqUzunlugu = loaded_graph.get_tensor_by_name('target_sequence_length:0')
    keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
    translate_logits = sess.run(logitler, {input_data: [cumle]*batchBuyuklugu, hedefSeqUzunlugu: [len(cumle)*2]*batchBuyuklugu, keep_prob: 1.0})[0]

print('Input')
print('  ID\'ler:      {}'.format([i for i in cumle]))
print('  Ingilizce Kelimeler: {}'.format([kaynakSozlugu[i] for i in cumle]))

print('\nTahmin')
print('  ID\'ler:      {}'.format([i for i in translate_logits]))
print('  Almanca Kelimeler: {}'.format(" ".join([hedefSozlugu[i] for i in translate_logits])))