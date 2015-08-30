
"""
This example trains a sequence2sequence network to translate europarl Spanish-English.
It's not particularly good yet, but it does work and is a good reference for sequence2sequence.

Training set example:
Iteration 134300: 1.77240937874
input:   en la 22a sesion , celebrada el 25 de octubre , el secretario de la comision senalo a la atencion de los presentes una nota
output:  at the 25th meeting , on 19 october , the chairman , the representative the statement to a statement by the records see a EOS
target:  the 22nd meeting , on 25 october , the secretary of the committee drew attention to a note by the secretariat ( a / UNK
"""
import numpy as np
import json
import random
import apollocaffe
from apollocaffe.layers import (Concat, LstmUnit, Dropout, Wordvec, NumpyData,
    Softmax, SoftmaxWithLoss, Filler, CapSequence, InnerProduct)


def get_data(data_config):
    # You can download this file with ...
    source = "%s/%s" % (data_config["data_prefix"], data_config["data_source"])
    target = "%s/%s" % (data_config["data_prefix"], data_config["data_target"])
    epoch = 0
    while True:
        with open(source, 'r') as f1:
            with open(target, 'r') as f2:
                l = zip(f1.readlines(), f2.readlines())
                if "data_samples" in data_config:
                    l = l[:data_config["data_samples"]]
                random.shuffle(l)
                for x, y in l:
                    if len(x.strip()) == 0 or len(y.strip()) == 0:
                        continue
                    x_processed = map(int, x.strip().split(' '))
                    y_processed = map(int, y.strip().split(' '))
                    yield (x_processed, y_processed)
        print("epoch %s finished" % epoch)
        epoch += 1

def unpad(sentence, net_config):
    result = []
    for x in sentence:
        if x == net_config["eos_symbol"]:
            break
        result.append(x)
    return result

def padded_reverse(sentence, net_config):
    try:
        result = sentence[:sentence.index(net_config["pad_symbol"])][::-1]
    except:
        result = sentence[::-1]
    return result
    
def pad_batch(sentence_batch, net_config):
    source_len = min(max(len(x) for x, y in sentence_batch), net_config["s_max_len"])
    target_len = min(max(len(y) for x, y in sentence_batch), net_config["t_max_len"])
    f_result = []
    b_result = []
    for x, y in sentence_batch:
        x_clip = x[:source_len]
        y_clip = y[:target_len]
        f_result.append(x_clip[::-1] + [net_config["pad_symbol"]] * (source_len - len(x_clip)))
        b_result.append(y_clip + [net_config["pad_symbol"]] * (target_len - len(y_clip)))
    return np.array(f_result), np.array(b_result)
    
def get_data_batch(config):
    net_config = config["net"]
    data_iter = get_data(config["data"])
    while True:
        raw_batch = []
        for i in range(net_config["batch_size"]):
            raw_batch.append(next(data_iter))
        sentence_batch = pad_batch(raw_batch, net_config)
        yield sentence_batch

def lstm_layers(name, mem_cells, step, batch_size, bottoms,
                dropout_ratio=0.0, seed=None, init_range=0.1):
    layers = []
    filler = Filler("uniform", init_range)
    if step == 0:
        prev_mem = name + ":mem_seed"
        layers.append(NumpyData(prev_mem, np.zeros((batch_size, mem_cells))))
        if seed is not None:
            prev_out = seed
        else:
            prev_out = name + ":seed"
            layers.append(NumpyData(prev_out,
                data=np.zeros((batch_size, mem_cells))))
    else:
        prev_out = name + ":out%d" % (step - 1)
        prev_mem = name + ":mem%d" % (step - 1)
    layers.append(Concat(name + ":concat%d" % step,
        bottoms=([prev_out] + bottoms)))
    lstm_tops = [name + ":out%d" % step, name + ":mem%d" % step]
    layers.append(LstmUnit(name + str(step), mem_cells,
        param_names=[name + x for x in
            [":input_value", ":input_gate", ":forget_gate", ":output_gate"]],
        bottoms=[name + ":concat%d" % step, prev_mem],
        tops=lstm_tops, weight_filler=filler))
    if dropout_ratio > 0.0:
        layers.append(Dropout(name + ":dropout%d" % step, dropout_ratio,
            bottoms=[name + ":out%d" % step],
            tops=[name + ":dropout%d" % step]))
    return layers

def softmax_choice(data):
    probs = data.flatten().astype(np.float64)
    probs /= probs.sum()
    return np.random.choice(range(len(probs)), p=probs)

def forward(net, net_config, sentence_batches, deploy=False):
    net.clear_forward()
    source_batch, target_batch = next(sentence_batches)

    filler = Filler("uniform", net_config["init_range"])
    net.f(NumpyData("source_lstm_seed",
        data=np.zeros((net_config["batch_size"], net_config["mem_cells"]))))
    lengths = [min(len([1 for token in x if token != net_config["pad_symbol"]]), net_config["s_max_len"]) for x in source_batch]
    
    # Thang Aug15
    softmax_bias = False
    num_layers = 2
    hidden_bottoms = ["source_lstm%d:seed" % (num_layers-1)]
    for step in range(source_batch.shape[1]):
        s = str(step)
        word = source_batch[:, step]
        net.f(NumpyData("source_word" + s, word))
        net.f(Wordvec("source_wordvec" + s, net_config["mem_cells"],
            net_config["vocab_size"], bottoms=["source_word" + s],
            param_names=["source_wordvec_param"], weight_filler=filler))

        # Thang Aug15
        prev_layer_str = "source_wordvec"
        layers = []
        for l in xrange(num_layers):
            next_layer_str = "source_lstm%d" % l 
            layers += lstm_layers(next_layer_str, net_config["mem_cells"],
                step, batch_size=net_config["batch_size"], bottoms=[prev_layer_str + s],
                init_range=net_config["init_range"])
            prev_layer_str = "%s:out" % next_layer_str

        for layer in layers:
            net.f(layer)

        # Thang Aug15
        hidden_bottoms.append("%s%d" % (prev_layer_str, step))

    net.f(CapSequence("hidden_seed", sequence_lengths=lengths,
        bottoms=hidden_bottoms))
    
    loss = []

    for step in range(target_batch.shape[1]):
        s = str(step)
        if step == 0:
            word = np.zeros((net_config["batch_size"], 1))
        else:
            if deploy:
                top = 'softmax%d' % (step - 1)
                word = [[softmax_choice(x)] for x in net.blobs[top].data]
            else:
                word = target_batch[:, step - 1]
        net.f(NumpyData("target_word" + s, word))
        net.f(Wordvec("target_wordvec" + s, net_config["mem_cells"],
            net_config["vocab_size"], bottoms=["target_word" + s],
            param_names=["target_wordvec_param"], weight_filler=filler))

        layers = lstm_layers("target_lstm", net_config["mem_cells"],
            step, batch_size=net_config["batch_size"], bottoms=["target_wordvec" + s],
            init_range=net_config["init_range"], seed="hidden_seed",
            dropout_ratio=0.15)
        for layer in layers:
            net.f(layer)

        net.f(NumpyData("label" + s,
            data=np.reshape(target_batch[:, step], (net_config["batch_size"], 1))))

        # Thang Aug15
        if softmax_bias: # with bias
          net.f(InnerProduct("ip" + s, bottoms=["target_lstm:dropout" + s],
              bias_term=softmax_bias, param_names=["ip_weight", "ip_bias"],
              num_output=net_config["vocab_size"], weight_filler=filler))
        else: # no bias
          net.f(InnerProduct("ip" + s, bottoms=["target_lstm:dropout" + s],
              bias_term=softmax_bias, param_names=["ip_weight"],
              num_output=net_config["vocab_size"], weight_filler=filler))
        loss.append(net.f(SoftmaxWithLoss("softmax_loss" + s, ignore_label=net_config["pad_symbol"],
            bottoms=["ip" + s, "label" + s])))
        loss.append(net.f(Softmax("softmax" + s,
            bottoms=["ip" + s])))
    return np.mean(loss)

def load_vocab(config):
    data_config = config["data"]
    import pickle
    t_vocab = config["data"]["t_vocab"]
    if t_vocab:
        with open("%s/%s" % (data_config["data_prefix"], t_vocab), 'r') as f:
            t_vocab = pickle.load(f)
    else:
        t_vocab = {chr(i): i for i in range(3, 256)}
    s_vocab_file = config["data"]["s_vocab"]
    if s_vocab_file:
        with open("%s/%s" % (data_config["data_prefix"], s_vocab_file), 'r') as f:
            s_vocab = pickle.load(f)
    else:
        s_vocab = {chr(i): i for i in range(3, 256)}
    t_ivocab_file = config["data"]["t_ivocab"]
    if t_ivocab_file:
        with open("%s/%s" % (data_config["data_prefix"], t_ivocab_file), 'r') as f:
            t_ivocab = pickle.load(f)
    else:
        t_ivocab = {i: chr(i) for i in range(3, 256)}
    s_ivocab_file = config["data"]["s_ivocab"]
    if s_ivocab_file:
        with open("%s/%s" % (data_config["data_prefix"], s_ivocab_file), 'r') as f:
            s_ivocab = pickle.load(f)
    else:
        s_ivocab = {i: chr(i) for i in range(3, 256)}

    for ivocab in s_ivocab, t_ivocab:
        assert 1 not in ivocab
        assert 2 not in ivocab
        ivocab[0] = "UNK"
        ivocab[1] = "EOS"
        ivocab[2] = "PAD"

    return s_vocab, s_ivocab, t_vocab, t_ivocab
def train(config):
    net = apollocaffe.ApolloNet()

    sentence_batches = get_data_batch(config)

    net_config = config["net"]
    forward(net, net_config, sentence_batches)
    solver = config["solver"]
    if solver["weights"]:
        net.load(solver["weights"])
    train_loss_hist = []
    #net.draw_to_file("/tmp/mt.jpg")

    s_vocab, s_ivocab, t_vocab, t_ivocab = load_vocab(config)

    logging = config["logging"]
    loggers = [
        apollocaffe.loggers.TrainLogger(logging["display"]),
        apollocaffe.loggers.TestLogger(solver["test_interval"]),
        apollocaffe.loggers.SnapshotLogger(logging["snapshot_interval"],
            logging["snapshot_prefix"]),
        ]
    for i in range(solver['start_iter'], solver["max_iter"]):
        train_loss_hist.append(forward(net, net_config, sentence_batches))
        net.backward()
        lr = (solver["base_lr"] * (solver["gamma"])**(i // solver["stepsize"]))
        net.update(lr=lr, clip_gradients=solver["clip_gradients"])
        if i > 2000 and i % 3000 == 0:
            config["net"]["s_max_len"] += 1
            config["net"]["t_max_len"] += 1
        if i % logging["display"] == 0:
            forward(net, net_config, sentence_batches, deploy=True)
            print("Iteration %d: %s" % (i, np.mean(train_loss_hist[-logging["display"]:])))
            output = []
            target = []
            source = []
            for step in range(net_config["s_max_len"]):
                try:
                    source.append(int(net.blobs["source_word%d" % step].data[0].flatten()[0]))
                except:
                    break
            for step in range(net_config["t_max_len"]):
                try:
                    output.append(np.argmax(net.blobs["softmax%d" % step].data[0].flatten()))
                    target.append(np.int(net.blobs["label%d" % step].data[0].flatten()))
                except:
                    break
            try:
                char = str(logging["split_output"])
                print("input:\t" + char.join(
                    s_ivocab[x] for x in padded_reverse(source, net_config)))
                print("output:\t" + char.join(
                    t_ivocab[x] for x in unpad(output, net_config)))
                print("target:\t" + char.join(
                    t_ivocab[x] for x in unpad(target, net_config)))
            except Exception as e:
                print e
        for logger in loggers:
            logger.log(i, {"train_loss": train_loss_hist,
                "apollo_net": net, "start_iter": 0})

def main():
    parser = apollocaffe.base_parser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    config = json.load(open(args.config, 'r'))
    if args.weights:
        config["solver"]["weights"] = args.weights
    config["solver"]["start_iter"] = args.start_iter

    apollocaffe.set_random_seed(config["solver"]["random_seed"])
    apollocaffe.set_device(args.gpu)
    apollocaffe.set_cpp_loglevel(args.loglevel)

    train(config)

if __name__ == "__main__":
    main()
