import numpy as np
import tensorflow as tf
from tensorflow.python.layers.core import Dense

#Hyperparameters
sozlukBuyuklugu = 50000
cumleSayisi = 50000
batchBuyuklugu = 128
rnnBoyutu = 128
layerSayisi = 3
epochSayisi = 13
encEmbeddingBoyutu = 128 
decEmbeddingBoyutu = 128
learningRate = 0.001
keepProbability = 0.5
dispAdim = 300

def cumleOku(dosyaAdi, sayi):
    cumleListesi = []
    dosya = open(dosyaAdi, encoding='utf-8')
    for kayit in dosya:
        cumleListesi.append(kayit.lower())
        if len(cumleListesi) >= sayi:
            return cumleListesi

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

def indisAl(tokenlar, sozluk):
    vektor = []
    for tok in tokenlar:
        vektor.append(sozluk[tok])
    return vektor

def cumledenVektoreCevir(kaynakCumleler, hedefCumleler, kaynakSozlugu, hedefSozlugu):
    kaynakEgitimCumleleri = []
    hedefEgitimCumleleri = []
    for i in range(cumleSayisi):
        kaynakTokenlar = tokenlaraAyir(kaynakCumleler[i],kaynakSozlugu)
        hedefTokenlar = tokenlaraAyir(hedefCumleler[i],hedefSozlugu)
        #Sozlukteki indis karsiliklarini elde etme
        kaynakCumleIndis = indisAl(kaynakTokenlar, kaynakSozlugu)
        hedefCumleIndis = indisAl(hedefTokenlar, hedefSozlugu)
        hedefCumleIndis.append(hedefSozlugu["<EOS>"])

        kaynakEgitimCumleleri.append(kaynakCumleIndis)
        hedefEgitimCumleleri.append(hedefCumleIndis)

    return kaynakEgitimCumleleri, hedefEgitimCumleleri

def batchePadEkle(batch, padBoyutu):
    maxCumle = max([len(cumle) for cumle in batch])
    return [cumle + [padBoyutu] * (maxCumle - len(cumle)) for cumle in batch]

def batchAl(kaynakEgitimCumleleri, hedefEgitimCumleleri, batchBuyuklugu, kaynakPadBoyutu, hedefPadBoyutu):
    for batch_i in range(0, len(kaynakEgitimCumleleri)//batchBuyuklugu):
        start_i = batch_i * batchBuyuklugu

        kaynakBatch = kaynakEgitimCumleleri[start_i:start_i + batchBuyuklugu]
        hedefBatch = hedefEgitimCumleleri[start_i:start_i + batchBuyuklugu]

        padKaynakBatch = np.array(batchePadEkle(kaynakBatch, kaynakPadBoyutu))
        padHedefBatch = np.array(batchePadEkle(hedefBatch, hedefPadBoyutu))

        padKaynakUzunlugu = []
        for kaynak in padKaynakBatch:
            padKaynakUzunlugu.append(len(kaynak))

        padHedefUzunlugu = []
        for hedef in padHedefBatch:
            padHedefUzunlugu.append(len(hedef))

        yield padKaynakBatch, padHedefBatch, padKaynakUzunlugu, padHedefUzunlugu

def accuracyAl(hedef, logitler):
    maxSeq = max(hedef.shape[1], logitler.shape[1])
    if maxSeq - hedef.shape[1]:
        hedef = np.pad(hedef,[(0,0),(0,maxSeq - hedef.shape[1])],'constant')
    if maxSeq - logitler.shape[1]:
        logits = np.pad(logitler,[(0,0),(0,maxSeq - logitler.shape[1])],'constant')
    return np.mean(np.equal(hedef, logitler))

def encDecModelGirdileri():
    girdiler = tf.placeholder(tf.int32, [None, None], name='input') #ilki batch size ikincisi max cümle uzunluğu [none none]
    hedefler = tf.placeholder(tf.int32, [None, None], name='targets') #ilki batch size ikincisi max cümle uzunluğu [none none]
    
    hedefSeqUzunlugu = tf.placeholder(tf.int32, [None], name='target_sequence_length') #helper'ın kullanması için
    maxHedefUzunlugu = tf.reduce_max(hedefSeqUzunlugu)    
    
    return girdiler, hedefler, hedefSeqUzunlugu, maxHedefUzunlugu

def hyperparamGirdileri():
    learningRate = tf.placeholder(tf.float32, name='lr_rate')
    keepProbability = tf.placeholder(tf.float32, name='keep_prob')
    return learningRate, keepProbability

def decoderInputuHazirla(hedefVeri,hedefSozlugu,batchBuyuklugu):
    goDegeri = hedefSozlugu["<GO>"]
    temp = tf.strided_slice(hedefVeri,[0, 0],[batchBuyuklugu, -1],[1, 1])
    sonInput = tf.concat([tf.fill([batchBuyuklugu,1],goDegeri),temp],1)
    return sonInput

def encodingKatmani(rnnGirdileri, rnnBoyutu, layerSayisi, keepProbability, kaynakSozlukBuyuklugu, encEmbeddingBoyutu):
    embed = tf.contrib.layers.embed_sequence(rnnGirdileri, vocab_size=kaynakSozlukBuyuklugu, embed_dim=encEmbeddingBoyutu)
    stackedCells = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(rnnBoyutu), keepProbability) for _ in range(layerSayisi)])   
    ciktilar, durum = tf.nn.dynamic_rnn(stackedCells, embed, dtype=tf.float32)
    return ciktilar, durum

def decodingEgitimKatmani(encoderDurumu, decoderCell, decEmbeddingGirdisi, hedefSeqUzunlugu, maxSmrUzunlugu, ciktiKatmani, keepProbability):
    decoderCell = tf.contrib.rnn.DropoutWrapper(decoderCell, output_keep_prob=keepProbability)
    helper = tf.contrib.seq2seq.TrainingHelper(decEmbeddingGirdisi, hedefSeqUzunlugu)   
    decoder = tf.contrib.seq2seq.BasicDecoder(decoderCell, helper, encoderDurumu, ciktiKatmani)
    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=True, maximum_iterations=maxSmrUzunlugu)
    return outputs

def decodingInferansKatmani(encoderDurumu, decoderCell, decEmbeddingGirdisi, seqIdBaslangici,seqIdBitisi, maxHedefSeqUzunlugu,
                            sozlukBuyuklugu, ciktiKatmani, batchBuyuklugu, keepProbability):
    decoderCell = tf.contrib.rnn.DropoutWrapper(decoderCell, output_keep_prob=keepProbability)
    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decEmbeddingGirdisi, tf.fill([batchBuyuklugu], seqIdBaslangici), seqIdBitisi)
    decoder = tf.contrib.seq2seq.BasicDecoder(decoderCell, helper, encoderDurumu, ciktiKatmani)
    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=True, maximum_iterations=maxHedefSeqUzunlugu)
    return outputs

def decodingKatmani(decGirdisi, encoderDurumu,hedefSeqUzunlugu, maxHedefSeqUzunlugu,rnnBoyutu,
                   layerSayisi, hedefSozlugu,batchBuyuklugu, keepProbability, decEmbeddingBoyutu):

    hedefSozlukUzunlugu = len(hedefSozlugu)
    decoderEmbedding = tf.Variable(tf.random_uniform([hedefSozlukUzunlugu, decEmbeddingBoyutu]))
    decoderEmbeddingGirdisi = tf.nn.embedding_lookup(decoderEmbedding, decGirdisi)
    
    cells = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(rnnBoyutu) for _ in range(layerSayisi)])
    
    with tf.variable_scope("decode"):
        ciktiKatmani = tf.layers.Dense(hedefSozlukUzunlugu)
        egitimCiktisi = decodingEgitimKatmani(encoderDurumu, cells, decoderEmbeddingGirdisi, hedefSeqUzunlugu, 
                                            maxHedefSeqUzunlugu, ciktiKatmani, keepProbability)

    with tf.variable_scope("decode", reuse=True):
        inferansCiktisi = decodingInferansKatmani(encoderDurumu, cells, decoderEmbeddingGirdisi, hedefSozlugu['<GO>'], hedefSozlugu['<EOS>'], 
                                            maxHedefSeqUzunlugu, hedefSozlukUzunlugu, ciktiKatmani, batchBuyuklugu, keepProbability)

    return (egitimCiktisi, inferansCiktisi)

def seq2seqModel(girdiVerisi,hedefVeri, keepProbability, batchBuyuklugu,hedefSeqUzunlugu,maxHedefCumleUzunlugu,
                  kaynakSozlukBoyutu, hedefSozlukBoyutu,encEmbeddingBoyutu, decEmbeddingBoyutu,rnnBoyutu, layerSayisi, hedefSozlugu):
    encCiktilari, encDurumlari = encodingKatmani(girdiVerisi, rnnBoyutu, layerSayisi, keepProbability, kaynakSozlukBoyutu, encEmbeddingBoyutu)
    decGirdisi = decoderInputuHazirla(hedefVeri,hedefSozlugu,batchBuyuklugu)
    egitimCiktisi, inferansCiktisi = decodingKatmani(decGirdisi,encDurumlari,hedefSeqUzunlugu,maxHedefCumleUzunlugu,
                                                    rnnBoyutu,layerSayisi,hedefSozlugu,batchBuyuklugu,keepProbability,decEmbeddingBoyutu)
    
    return egitimCiktisi, inferansCiktisi

kaynakSozlugu = sozlukOku("vocab.50K.en.txt")
hedefSozlugu = sozlukOku("vocab.50K.de.txt")

tersKaynakSozlugu = dict(zip(kaynakSozlugu.values(), kaynakSozlugu.keys()))
tersHedefSozlugu = dict(zip(hedefSozlugu.values(), hedefSozlugu.keys()))

kaynakCumleler = cumleOku("train.en", cumleSayisi)
hedefCumleler = cumleOku("train.de", cumleSayisi)

maxKaynakCumleUzunlugu = max([len(sentence.split(" ")) for sentence in kaynakCumleler])
maxHedefCumleUzunlugu = max([len(sentence.split(" ")) for sentence in hedefCumleler])

kaynakEgitimCumleleri, hedefEgitimCumleleri = cumledenVektoreCevir(kaynakCumleler, hedefCumleler, kaynakSozlugu,hedefSozlugu)

egitimGraph = tf.Graph()
with egitimGraph.as_default():
    girdiVerisi, hedefler, hedefSeqUzunlugu, maxHedefSeqUzunlugu = encDecModelGirdileri()
    learningRate, keepProbability = hyperparamGirdileri()
    egitimLogit, inferansLogit = seq2seqModel(tf.reverse(girdiVerisi, [-1]), hedefler, keepProbability, batchBuyuklugu, hedefSeqUzunlugu,
                                                    maxHedefSeqUzunlugu, len(kaynakSozlugu),len(hedefSozlugu), encEmbeddingBoyutu,decEmbeddingBoyutu,
                                                    rnnBoyutu,layerSayisi,hedefSozlugu)
    
    egitimLogitleri = tf.identity(egitimLogit.rnn_output, name='logits')
    inferansLogitleri = tf.identity(inferansLogit.sample_id, name='predictions')
    masklar = tf.sequence_mask(hedefSeqUzunlugu, maxHedefSeqUzunlugu, dtype=tf.float32, name='masks')

    with tf.name_scope("optimization"):
        cost = tf.contrib.seq2seq.sequence_loss(egitimLogitleri,hedefler, masklar)
        optimizer = tf.train.AdamOptimizer(learningRate)
        # Gradient Clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)


kaynakEgitim = kaynakCumleler[batchBuyuklugu:]
hedefEgitim = hedefCumleler[batchBuyuklugu:]
kaynakDogrulama = kaynakCumleler[:batchBuyuklugu]
hedefDogrulama = hedefCumleler[:batchBuyuklugu]
(kaynakDogrulamaBatch, hedefDogrulamaBatch, kaynakDogrulamaUzunluk, hedefDogrulamaUzunluk) = next(batchAl(kaynakdogrulama,
                                                                                                             hedefDogrulama,
                                                                                                             batchBuyuklugu,
                                                                                                             kaynakSozlugu['<PAD>'],
                                                                                                             hedefSozlugu['<PAD>']))                                                                                                  
with tf.Session(graph=train_graph) as session:
    session.run(tf.global_variables_initializer())

    for epoch in range(epochSayisi):
        for batch, (kaynakBatch,hedefBatch,kaynakUzunluklari, hedefUzunluklari) in enumerate(
                batchAl(kaynakEgitim, hedefEgitim, batchBuyuklugu, kaynakSozlugu['<PAD>'], hedefSozlugu['<PAD>'])):

            _, loss = session.run([train_op, cost], {input_data: kaynakBatch, targets: hedefBatch, lr: learningRate,
                                    target_sequence_length: hedefUzunluklari, keep_prob: keepProbability})

            if batch % dispAdim == 0 and batch > 0:
                egitimBatchLogitleri = session.run(inferansLogitleri,{input_data: kaynakBatch,target_sequence_length: hedefUzunluklari,keep_prob: 1.0})
                dogrulamaBatchLogitleri = session.run(inferansLogitleri,{input_data: kaynakDogrulamaBatch,target_sequence_length: hedefUzunluklari, keep_prob: 1.0})
                trainAccuracy = accuracyAl(hedefBatch, egitimBatchLogitleri)
                validAccuracy = accuracyAl(hedefDogrulamaBatch, dogrulamaBatchLogitleri)
                print('Epoch {:>3} Batch {:>4}/{} - Train Accuracy: {:>6.4f}, Validation Accuracy: {:>6.4f}, Loss: {:>6.4f}'
                      .format(epoch, batch, len(kaynakCumleler) // batchBuyuklugu, trainAccuracy, validAccuracy, loss))

    saver = tf.train.Saver()
    saver.save(session, "\kayit")
    print('Model kaydedildi')