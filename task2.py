import numpy as np
from display import Display


class Neuron:
    def __init__(self, no_features= 2, activation= None):
        self.weights = np.random.rand(no_features + 1)
        self.derivative = {
            self.__class__.heaviside: lambda x: 1,
            self.__class__.sigmoid: lambda x: self.__class__.sigmoid(x) *  (1 - self.__class__.sigmoid(x)),
            np.sin: np.cos,
            np.tanh: lambda x: 1-np.power(np.tanh(x), 2),
            np.sign: lambda x: 1,
            self.__class__.relu: lambda x: (x > 0) * 1,
            self.__class__.leaky_relu: lambda x: 1 if x > 0 else 0.01,
        }
        if not activation:
            self.activation = self.__class__.heaviside
        else:
            if activation not in self.derivative.keys():
                print(f'activation should be one of {[map(str ,self.derivative.keys())]}')
                raise KeyError
            self.activation = activation
    def forward(self, x):
        x = np.concatenate((np.ones(1), x))
        return self.activation(np.dot(x, self.weights))
    def train(self, x, y, y_prime):
        x = np.concatenate((np.ones(1), x))
        learning_rate = 0.05
        adj = learning_rate * (y_prime - y) * self.derivative[self.activation](self.weights.T @ x) * x
        self.weights += adj
    def test(self, test_data, labels):
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for x, y_prime in zip(test_data, labels):
            y = round(self.forward(x))
            # print(y, y_prime)
            if y_prime:
                if y: tp += 1
                else: fn += 1
            else:
                if y: fp += 1
                else: tn += 1
            # print(f'{tp=}, {tn=}, {fp=}, {fn=}')
        try:
            accuracy = (tp + tn) / len(test_data)#(tp + tn + fp + fn)
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
        except ZeroDivisionError:
            accuracy = 1
            precision = 1
            recall = 1
            print(f'{tp=}, {tn=}, {fp=}, {fn=}')
        return accuracy, precision, recall



    
    def heaviside(x):
        return (x > 0).astype('float')
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    def relu(x):
        return x * (x > 0)
    def leaky_relu(x):
        return np.maximum(x, 0.001 * x)
        

def normalize(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
def generate_single_mode(no_samples):
    rng = np.random.default_rng()
    mean = rng.random() * 20 - 10
    variance = rng.random()
    data = rng.normal(mean, variance, (no_samples, 2))
    return data

def generate_data(no_samples):
    
    training = int(2*0.8*no_samples)        
    a = generate_single_mode(no_samples)
    b = generate_single_mode(no_samples)

    
    a_b = np.concatenate((a,b))
    a_b = normalize(a_b)

    first_set = a_b[:len(a_b)//2]
    second_set = a_b[len(a_b)//2:]

    a = np.c_[a, np.zeros(no_samples)]
    b = np.c_[b, np.ones(no_samples)]

    c = np.concatenate((a,b))

    c[:,0] = normalize(c[:,0])
    c[:,1] = normalize(c[:,1])
    np.random.shuffle(c)

    train_samples, train_labels = c[:training, :2], c[:training, 2]
    test_samples, test_labels = c[:training, :2], c[:training, 2]

    return train_samples, train_labels, test_samples, test_labels, first_set, second_set



n = Neuron()

no_samples = 100

train_samples, train_labels, test_samples, test_labels, first_set, second_set = generate_data(no_samples)

# for sample, label in zip(train_samples, train_labels):
#     res = n.forward(sample)
#     n.train(sample, res, label)
# print(n.test(test_samples, test_labels))
# print(n.weights)

samples_gen = iter(zip(train_samples, train_labels))

def on_click(no_samples, display: Display):
    try:
        for i in range(no_samples.get()):
            next_sample, next_label = next(samples_gen)
            res = n.forward(next_sample)
            n.train(next_sample, res, next_label)
        print(n.weights)
        display.display_decision_boundary(n.weights)
    except StopIteration:
        Display.show_end_of_samples_msgbox()


d = Display(on_click)
d.display_data(first_set, second_set)
d.display_decision_boundary(n.weights)
d.show()


