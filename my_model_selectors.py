import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    ''' 
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        # TODO implement model selection based on BIC scores
        best_model = None
        best_bic = 100000000000000

        n = len(self.lengths)
        logN = np.log(n)

        for n_comp in range(self.min_n_components, self.max_n_components+1):
            try:
                model = GaussianHMM(n_components=n_comp, n_iter=1000, random_state=self.random_state).fit(self.X, self.lengths)
                #  BIC =−2logL+plogN 
                #  where L is the likelihood of the fitted model, p is the number of parameters, and N is the number of data points.
                logL = model.score(self.X, self.lengths)
                p = n**2 + 2 * n_comp * n - 1
                curr_bic = -2 * logL + p * logN
                if curr_bic < best_bic:
                    best_bic, best_model = curr_bic, model
            except:
                continue


        return best_model



class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        max_dist = -10000000000000000000
        best_model = None

        for n_comp in range(self.min_n_components, self.max_n_components+1):
            model = None
            logL = -100000000000000000
            log_sum = 0
            count = 0
            try:
                model = GaussianHMM(n_components=n_comp, n_iter=1000, random_state=self.random_state).fit(self.X, self.lengths  )
                logL = model.score(self.X, self.lengths)
            except:
                continue

            for w in self.hwords:
                if w != self.this_word:
                    log_sum += model.score(self.hwords[w][0], self.hwords[w][1])
                    count += 1
                    log_avg = log_sum/count*1.0

            if logL - log_avg > max_dist:
                best_model, max_dist = model, logL - log_avg

        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        best_model = None
        best_logL = -100000000000000

        n_splits = min(3, len(self.sequences))
        if n_splits < 2:
            return None

        split_method = KFold(n_splits=n_splits)


        for n_comp in range(self.min_n_components, self.max_n_components+1):
            count = 0
            logL_count = 0
            model = GaussianHMM(n_components=n_comp, n_iter=1000, random_state=self.random_state)

            for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                X_train, lengths_train = combine_sequences(cv_train_idx, self.sequences)
                X_test, lengths_test = combine_sequences(cv_test_idx, self.sequences)

                try:
                    logL_count += model.fit(X_train, lengths_train).score(X_test, lengths_test)
                    count += 1
                except:
                    continue

            if count > 0:
                mean_logL = logL_count/float(count)
            else:
                mean_logL = -100000000000000

            if mean_logL > best_logL:
                best_model, best_logL = model, mean_logL


        return best_model
