import math
import numpy as np
import tensorflow as tf
import os

sozlukBoyutu = 50000
nodeSayisi = 128
inputUzunlugu = 128
batchBoyutu = 16
kaynakSeqUzunlugu = 40
hedefSeqUzunlugu = 60
cumleSayisi = 50000
maxKaynakCumleUzunlugu = 41 #Ornekteki ortalamalara göre secilmis
maxHedefCumleUzunlugu = 61 #Ornekteki ortalamalara göre secilmis
adimSayisi = 10000
ortalamaKayip = 0.0

def sozlukOku(sozlukAdi):
    sozluk = {}
    dosya = open(sozlukAdi,encoding="utf-8")
    for kayit in dosya:
        sozluk[kayit[:-1]] = len(sozluk)
    return sozluk

def cumleOku(dosyaAdi,sayi):
    cumleListesi = []
    dosya = open(dosyaAdi, encoding='utf-8')
    for kayit in dosya:
        cumleListesi.append(kayit)
        if len(cumleListesi)>=sayi:
            return cumleListesi    

def tokenlaraAyir(cumle,sozluk):
    cumle = cumle.replace(',',' ,').replace('.',' .').replace('\n',' ')
    tokenlar = cumle.split(' ')
    tokY = []
    for tok in tokenlar:
        if tok not in sozluk.keys():
            tokY.append("<unk>")
        else:
            tokY.append(tok)
    return tokY

def vektorAl(tokenlar,sozluk):
    vektor = []
    for tok in tokenlar:
        vektor.append(sozluk[tok])
    return vektor

def uzunlukTamamla(vektor,maxUzunluk,sozluk):
    #Max'tan buyukse max, degilse max'a tamamlayacak kadar </s> token'i ekle
    if len(vektor)<maxUzunluk:
        vektor.extend([sozluk['</s>'] for _ in range(maxUzunluk - len(vektor))])
    elif len(vektor)>maxUzunluk:
        vektor = vektor[:maxUzunluk]
    return vektor

#Batch icin (batch boyutu adet cumlenin bir sonraki indeksindeki degerleri aliyor)
#Cursor kismini tekrar incele
class DataGeneratorMT(object):
    
    def __init__(self,batchBoyutu,num_unroll,is_source):
        self._batchBoyutu = batchBoyutu
        self._num_unroll = num_unroll
        self._cursor = [0 for offset in range(self._batchBoyutu)]     
        self._sent_ids = None   
        self._is_source = is_source       
                
    def sonrakiBatch(self, sent_ids, first_set):
        
        if self._is_source:
            maxCumleUzunlugu = maxKaynakCumleUzunlugu
        else:
            maxCumleUzunlugu = maxHedefCumleUzunlugu
        batch_labelind = []
        batch_data = np.zeros((self._batchBoyutu),dtype=np.float32)
        batch_labels = np.zeros((self._batchBoyutu),dtype=np.float32)
        
        for b in range(self._batchBoyutu):
            
            sent_id = sent_ids[b]
            
            if self._is_source:
                sent_text = kaynakEgitimCumleleri[sent_id]
                             
                batch_data[b] = sent_text[self._cursor[b]]
                batch_labels[b]=sent_text[self._cursor[b]+1]

            else:
                sent_text = hedefEgitimCumleleri[sent_id]
                
                batch_data[b] = sent_text[self._cursor[b]]
                batch_labels[b] = sent_text[self._cursor[b]+1]

            self._cursor[b] = (self._cursor[b]+1)%(maxCumleUzunlugu-1)
                                    
        return batch_data,batch_labels
        
    def unroll_batches(self,sent_ids):
        
        if sent_ids is not None:
            
            self._sent_ids = sent_ids
            
            self._cursor = [0 for _ in range(self._batchBoyutu)]
                
        unroll_data,unroll_labels = [],[]
        inp_lengths = None
        for ui in range(self._num_unroll):
            
            data, labels = self.sonrakiBatch(self._sent_ids, False)
                    
            unroll_data.append(data)
            unroll_labels.append(labels)
            inp_lengths = kaynakEgitimUzunluklari[sent_ids]
        return unroll_data, unroll_labels, self._sent_ids, inp_lengths
    
    def reset_indices(self):
        self._cursor = [0 for offset in range(self._batchBoyutu)]

#Sozluklerin okunmasi
kaynakSozlugu = sozlukOku("vocab.50K.de.txt")
hedefSozlugu = sozlukOku("vocab.50K.en.txt")

tersKaynakSozlugu = dict(zip(kaynakSozlugu.values(),kaynakSozlugu.keys()))
tersHedefSozlugu = dict(zip(hedefSozlugu.values(),hedefSozlugu.keys()))

#Cumlelerin okunmasi
kaynakCumleler = cumleOku("train.de",cumleSayisi)
hedefCumleler = cumleOku("train.en",cumleSayisi)

#Tokenlara ayirma
kaynakEgitimCumleleri = []
hedefEgitimCumleleri = []
kaynakEgitimUzunluklari = []
hedefEgitimUzunluklari = []

for i, (kaynakCumle, hedefCumle) in enumerate(zip(kaynakCumleler,hedefCumleler)):
    kaynakTokenlar = tokenlaraAyir(kaynakCumle,kaynakSozlugu)
    hedefTokenlar = tokenlaraAyir(hedefCumle,hedefSozlugu)

    #Sozlukteki indis karsiliklarini elde etme
    kaynakCumleNumerik = vektorAl(kaynakTokenlar,kaynakSozlugu)
    kaynakCumleNumerik = kaynakCumleNumerik[::-1] #Daha iyi sonuc vermesi için kaynak cumle tersine cevriliyor
    kaynakCumleNumerik.insert(0,kaynakSozlugu['<s>'])#Baslangic token'i

    kaynakEgitimUzunluklari.append(min(len(kaynakCumleNumerik)+1,maxKaynakCumleUzunlugu))#Max'tan buyukse max 
    kaynakCumleNumerik = uzunlukTamamla(kaynakCumleNumerik,maxKaynakCumleUzunlugu,kaynakSozlugu)
    kaynakEgitimCumleleri.append(kaynakCumleNumerik)

    hedefCumleNumerik = vektorAl(hedefTokenlar,hedefSozlugu)
    hedefCumleNumerik.insert(0,hedefSozlugu['</s>'])#</s> ile basla

    hedefEgitimUzunluklari.append(min(len(hedefCumleNumerik)+1,maxHedefCumleUzunlugu))
    hedefCumleNumerik = uzunlukTamamla(hedefCumleNumerik,maxHedefCumleUzunlugu,hedefSozlugu)
    hedefEgitimCumleleri.append(hedefCumleNumerik)

kaynakEgitimCumleleri = np.array(kaynakEgitimCumleleri, dtype=np.int32)
hedefEgitimCumleleri = np.array(hedefEgitimCumleleri, dtype=np.int32)
kaynakEgitimUzunluklari = np.array(kaynakEgitimUzunluklari, dtype=np.int32)
hedefEgitimUzunluklari = np.array(hedefEgitimUzunluklari, dtype=np.int32)

tf.reset_default_graph()

encKaynakEgitimCumleleri = []
decKaynakEgitimCumleleri = []

#Embedding vektörleri
encoderEmbKatmani = tf.convert_to_tensor(np.load('de-embeddings.npy'))
decoderEmbKatmani = tf.convert_to_tensor(np.load('en-embeddings.npy'))

#Kaynak cumle uzunlugu kadar batch uzunlugunda placeholder olusturma
for i in range(kaynakSeqUzunlugu):
    encKaynakEgitimCumleleri.append(tf.placeholder(tf.int32, shape=[batchBoyutu],name='enc_train_inputs_%d'%i))

decEgitimEtiketleri = []
decEtiketMaskeleri = []

#Hedef cumle uzunlugu kadar placeholder kaynak,egitim ve etiket icin
for i in range(hedefSeqUzunlugu):
    decKaynakEgitimCumleleri.append(tf.placeholder(tf.int32, shape=[batchBoyutu],name='dec_train_inputs_%d'%i))
    decEgitimEtiketleri.append(tf.placeholder(tf.int32, shape=[batchBoyutu],name='dec-train_outputs_%d'%i))
    decEtiketMaskeleri.append(tf.placeholder(tf.float32, shape=[batchBoyutu],name='dec-label_masks_%d'%i))

#src embedding vektorde neyi elde ediyor? 
#Boyutlar 40x16lık cumle icin 128 uzunlugunda vektorlere tutuyor
encoderEmbeddingInp = [tf.nn.embedding_lookup(encoderEmbKatmani, src) for src in encKaynakEgitimCumleleri]
encoderEmbeddingInp = tf.stack(encoderEmbeddingInp)

decoderEmbeddingInp = [tf.nn.embedding_lookup(decoderEmbKatmani, src) for src in decKaynakEgitimCumleleri]
decoderEmbeddingInp = tf.stack(decoderEmbeddingInp)

encKaynakEgitimUzunluklari = tf.placeholder(tf.int32, shape=[batchBoyutu],name='train_input_lengths')
decKaynakEgitimUzunluklari = tf.placeholder(tf.int32, shape=[batchBoyutu],name='train_output_lengths')

#ENCODER
encoderCell = tf.nn.rnn_cell.LSTMCell(nodeSayisi)
initialState = encoderCell.zero_state(batchBoyutu, dtype=tf.float32)
#Redundant pad'leri hesaplamaması için dynamic?
encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoderCell, encoderEmbeddingInp, initial_state=initialState,sequence_length=encKaynakEgitimUzunluklari, time_major=True, swap_memory=True)

#DECODER - Helper nasil calisiyor bak
decoder_cell = tf.nn.rnn_cell.LSTMCell(nodeSayisi)
projection_layer = tf.layers.Dense(units=sozlukBoyutu, use_bias=True)
helper = tf.contrib.seq2seq.TrainingHelper(decoderEmbeddingInp, [maxHedefCumleUzunlugu-1 for _ in range(batchBoyutu)], time_major=True)
decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, encoder_state,output_layer=projection_layer) #attention da olabilirmiş?
outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, output_time_major=True,swap_memory=True)

logits = outputs.rnn_output #son layer
crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=decEgitimEtiketleri, logits=logits)
loss = (tf.reduce_sum(crossent*tf.stack(decEtiketMaskeleri)) / (batchBoyutu*hedefSeqUzunlugu))
train_prediction = outputs.sample_id

#Gradient Clipping (Neden Gradient Clipping kullaniliyor)

global_step = tf.Variable(0, trainable=False)
inc_gstep = tf.assign(global_step,global_step + 1)
learning_rate = tf.train.exponential_decay(0.01, global_step, decay_steps=10, decay_rate=0.9, staircase=True)

#Adam optimizer (Egitim asamasinda ilk belirli bir sayida iterasyon icin adam daha sonra sgd var. Neden?)
with tf.variable_scope('Adam'):
    adam_optimizer = tf.train.AdamOptimizer(learning_rate)

adam_gradients, v = zip(*adam_optimizer.compute_gradients(loss))
adam_gradients, _ = tf.clip_by_global_norm(adam_gradients, 25.0)
adam_optimize = adam_optimizer.apply_gradients(zip(adam_gradients, v))

#SGD optimizer
with tf.variable_scope('SGD'):
    sgd_optimizer = tf.train.GradientDescentOptimizer(learning_rate)

sgd_gradients, v = zip(*sgd_optimizer.compute_gradients(loss))
sgd_gradients, _ = tf.clip_by_global_norm(sgd_gradients, 25.0)
sgd_optimize = sgd_optimizer.apply_gradients(zip(sgd_gradients, v))

session = tf.InteractiveSession()

#Egitim
bleuSkorlari = []
kayiplar = []
tf.global_variables_initializer().run()

#Batch almak icin 
encoderDataGenerator = DataGeneratorMT(batchBoyutu=batchBoyutu,num_unroll=kaynakSeqUzunlugu,is_source=True)
decoderDataGenerator = DataGeneratorMT(batchBoyutu=batchBoyutu,num_unroll=hedefSeqUzunlugu,is_source=False)

for adim in range(adimSayisi+1):
    print("Adim ", str(adim+1))
    randomIdLer = np.random.randint(low=0,high=kaynakEgitimCumleleri.shape[0],size=(batchBoyutu))
    #labels data batchinin bir sonraki indisteki degerleri
    encoderDatalar, encoderEtiketler, _, encoderUzunluklar = encoderDataGenerator.unroll_batches(sent_ids=randomIdLer)
    
    feed_dict = {}
    feed_dict[encKaynakEgitimUzunluklari] = encoderUzunluklar #uzunluk tensor'unu ekle
    for i,(cumle,_) in enumerate(zip(encoderDatalar,encoderEtiketler)):            
        feed_dict[encKaynakEgitimCumleleri[i]] = cumle #encKaynakEgitimCumleleri'ne karsilik gelen cumleleri sozluge doldur.
    
    decoderDatalar, decoderEtiketler, _, decoderUzunluklar = decoderDataGenerator.unroll_batches(sent_ids=randomIdLer)
    
    feed_dict[decKaynakEgitimUzunluklari] = decoderUzunluklar #decoder uzunluk tensor'unu ekle
    for i,(cumle,etiket) in enumerate(zip(decoderDatalar,decoderEtiketler)):            
        feed_dict[decKaynakEgitimCumleleri[i]] = cumle
        feed_dict[decEgitimEtiketleri[i]] = etiket
        feed_dict[decEtiketMaskeleri[i]] = (np.array([i for _ in range(batchBoyutu)])<decoderUzunluklar).astype(np.int32)
    
    #Optimizasyon (NEDEN 10000 adimi gectikten sonra SGD'ye donuyor arastir)
    if adim < 10000:
        _,l,tr_pred = session.run([adam_optimize,loss,train_prediction], feed_dict=feed_dict)
    else:
        _,l,tr_pred = session.run([sgd_optimize,loss,train_prediction], feed_dict=feed_dict)
    tr_pred = tr_pred.flatten()

    ortalamaKayip += l
    if (adim+1)%50==0:
        cikti = 'Cumle: '
        for w in np.concatenate(decoderEtiketler,axis=0)[::batchBoyutu].tolist():
            cikti += tersHedefSozlugu[w] + ' '                    
            if tersHedefSozlugu[w] == '</s>':
                break                      
        print(cikti)        
        cikti = 'Tahmin: '
        for w in tr_pred[::batchBoyutu].tolist():
            cikti += tersHedefSozlugu[w] + ' '
            if tersHedefSozlugu[w] == '</s>':
                break
        print(cikti)
        print("Kayip: ",ortalamaKayip/50.0)        
        kayiplar.append(ortalamaKayip/50.0)             
        ortalamaKayip = 0.0
        session.run(inc_gstep)