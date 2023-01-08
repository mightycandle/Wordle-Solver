import numpy as np
import string
from scipy.stats import zscore
from collections import defaultdict as dd

class WordleSolver:
    def __init__(self):
        self.dictionary_file_location = 'all_words.txt'
                
        self.full_dictionary = self.build_dictionary()
        self.full_ordered_dictionary = self.full_dictionary.copy()
        self.dictionary = self.full_dictionary.copy()
        
        self.alphabets = list(string.ascii_lowercase)
        
        self.full_grid = self.get_grid()
        self.grid = self.full_grid.copy()        
        
        self.n_grams = self.init_n_grams()
        self.n_grams_scores = self.get_n_grams_score()
        
        self.init_frequent_letters = self.get_frequent_letters()
        self.frequent_letters = self.init_frequent_letters.copy()
        
        self.init_alphabet_rank = self.get_alphabet_ranks()
        self.alphabet_rank = self.init_alphabet_rank.copy()
        
        self.init_alphabet_best_index = self.get_alphabet_best_index(2)
        self.alphabet_best_index = self.init_alphabet_best_index.copy()
        
        self.init_word_scores = self.get_word_scores()
        self.word_scores = self.init_word_scores.copy()
        
        self.init_word_index = self.get_word_index()
        self.word_index = self.init_word_index.copy()
        
        self.tries_left = 6
        self.break_point = [4, 2]
        self.limiter_count = 8
        
        self.guessed_words = []
        self.correct_letters = set()
        self.wrong_letters = set()

        self.answer = ''
        self.green = ['' for i in range(5)]
        self.yellow = [[] for i in range(5)]
        
        self.prev_guess = ''
        self.previous_best_words = []
        
    def reset_variables(self):
        if len(self.dictionary) != len(self.full_dictionary):
            self.dictionary = self.full_ordered_dictionary.copy()
            
            self.grid = self.full_grid.copy()
            
            self.frequent_letters = self.init_frequent_letters.copy()
            
            self.alphabet_rank = self.init_alphabet_rank.copy()
            self.alphabet_best_index = self.init_alphabet_best_index.copy()
            
            self.word_scores = self.init_word_scores.copy()
            self.word_index = self.init_word_index.copy()
            
            
        self.guessed_words = []
        self.correct_letters = set()
        self.wrong_letters = set()
        
        self.answer = ''
        self.green = ['' for i in range(5)]
        self.yellow = [[] for i in range(5)]        
        
        self.prev_guess = ''
        self.previous_best_words = []
        
        self.tries_left = 6

    def update_dictionary(self):
        new_dict = []
        for word in self.dictionary:
            if word in self.guessed_words:
                continue
            
            is_good_word = True
            for i in range(5):
                if word[i] in self.wrong_letters:
                    is_good_word = False
                    break
                if self.green[i] != '' and self.green[i] != word[i]:
                    is_good_word = False
                    break
                if word[i] in self.yellow[i]:
                    is_good_word = False
                    break
            
            if is_good_word:
                new_dict.append(word)   
        # print("Updated dict_size = ",len(new_dict))
        return new_dict
        
    def recalibrate(self):
        self.dictionary = self.update_dictionary()
        
        self.grid = self.get_grid()
        self.frequent_letters = self.get_frequent_letters()
        
        self.alphabet_rank = self.get_alphabet_ranks()
        self.alphabet_best_index = self.get_alphabet_best_index(2)
        
        self.word_scores = self.get_word_scores()
        self.word_index = self.get_word_index()
        
    def build_dictionary(self):
        text_file = open(self.dictionary_file_location,"r")
        full_dictionary = text_file.read().splitlines()
        text_file.close()
        return full_dictionary
    
    def get_hash(self, s):
        p = 31
        hash = 0
        for i in range(3):
            hash += (1 + (ord(s[i]) - ord('a')))*(p ** i)
    
        return hash
    
    def init_n_grams(self):
        mp = [[0 for i in range(3)] for i in range(26800)]
        
        for w in self.full_dictionary:
            for i in range(3):
                mp[self.get_hash(w[i:i+3])][i] += 1
        
        return mp
    
    def get_n_grams_score(self):
        scores = []
        
        for word in self.full_dictionary:
            cur_score = 1.0
            terms = []
            for i in range(3):
                h = self.get_hash(word[i:i+3])
                cur_score *= ((self.n_grams[h][i]/sum(self.n_grams[h])))
                terms.append((self.n_grams[h][i]/sum(self.n_grams[h])))
            scores.append(cur_score)
            
        mp = dd(float)
        
        scores = zscore(scores)

        for i, j in enumerate(self.full_dictionary):
            mp[j] = scores[i]
        return mp
    
    def get_grid(self):
        grid = [[0 for i in range(5)] for i in range(26)]
        
        for word in self.dictionary:
            for i in range(5):
                grid[ord(word[i]) - ord('a')][i] += 1

        return np.array(grid)        
            
    def get_frequent_letters(self):
        frequency = dd(int)
        for word in self.dictionary:
            for letter in set(word):
                frequency[letter] += 1

        letters = self.alphabets.copy()
        letters.sort(key = lambda x: -frequency[x])
        
        return letters
            
    def get_alphabet_ranks(self):
        mp = dd(str)
        for i in range(26):
            mp[self.frequent_letters[i]] = i + 1
        
        return mp
    
    def get_alphabet_best_index(self, toprank = 2):
        
        mp = dd(list)
        
        for letter in self.alphabets:
            freq = self.grid[ord(letter) - ord('a')]
            track = []
            for i in range(5):
                track.append([-freq[i], i])
            track.sort()
            
            for i in range(toprank):
                mp[letter].append(track[i][1])
        
        return mp
           
    def get_word_scores(self):
        mp = dd(float)
        for word in self.dictionary:
            mp[word] = self.get_score(word)
        
        
        if len(self.dictionary) == len(self.full_ordered_dictionary):
            if self.dictionary[0] != 'tares':
                self.dictionary.sort(key = lambda x: mp[x])
                
        else:
            self.dictionary.sort(key = lambda x: mp[x])
        
        if self.full_ordered_dictionary[0] != 'tares':
            self.full_ordered_dictionary.sort(key = lambda x: mp[x])
        
        
        return mp
    
    def get_word_index(self):
        mp = dd(int)
        for i, j in enumerate(self.dictionary):
            mp[j] = i
            
        return mp
    
    def get_score(self, word):
        cur_score = 0.0
        visited = [0 for i in range(26)]

        for i in range(5):
            letter = word[i]
            if visited[ord(letter) - ord('a')]:
                cur_score += 5 * self.alphabet_rank[letter]
            else:
                if i in self.alphabet_best_index[letter]:
                    cur_score += 1 * self.alphabet_rank[letter]
                else:
                    cur_score += 5 * self.alphabet_rank[letter]
            visited[ord(letter) - ord('a')] = 1
        
        unique_count = len(set(word))
        cur_score += (5 - unique_count) * 13

        return cur_score - (unique_count - 3.5)*self.n_grams_scores[word]
        
    def get_value(self, answer, guess):
        green = [0 for i in range(5)]
        
        for i in range(5):
            if answer[i] == guess[i]:
                green[i] = 1
                
        status = [0, 0, 0]
        for i in range(5):
            if green[i]:
                status[0] += 1
                
            elif guess[i] in answer:
                status[1] += 1
            
            else:
                status[2] += 1
        return status
            
    def get_first_word(self):
        good_letters = self.frequent_letters[:self.limiter_count + 1]
        
        good_words = []
        
        def add_by_good_letters(letter_count):
            for word in self.dictionary:
                count = 0
                for letter in good_letters:
                    if letter in word:
                        count += 1
                if count >= letter_count:
                    good_words.append(word)
        
        add_by_good_letters(4)
        
        if good_words == []:
            add_by_good_letters(3)
        
        if good_words == []:
            good_words = self.dictionary
        
        good_words.sort(key = lambda x: self.word_scores[x])
        
        mp = dd(str)
        
        for answer in good_words[:min(len(good_words), 500)]:
            cur = 0
            for word in self.dictionary:
                cur_stats = self.get_value(answer, word)
                
                cur += cur_stats[0] * (3)
                cur += cur_stats[1] * (2)
                cur += cur_stats[2] * (-1)
            
            mp[answer] = cur
        
        good_words.sort(key = lambda x: -mp[x])
        
        return good_words[0]
        
    def display(self, pred, answer):
        stats = [0 for i in range(5)]
        
        for i in range(5):
            if pred[i] not in answer:
                stats[i] = 0
                self.wrong_letters.add(pred[i])
            
            else:
                self.correct_letters.add(pred[i])
                
                if pred[i] != answer[i]:
                    stats[i] = 1
                    self.yellow[i].append(pred[i])
                else:
                    stats[i] = 2
                    self.green[i] = pred[i]
        
        # print("\nGuess #{} - {}".format(7 - self.tries_left, pred))
        # print(stats)
        # print(self.previous_best_words)
               
    def get_next_words(self, word):
        possible = []
        mp = dd(list)

        previously_processed = dd(int)
        for prev_word in self.previous_best_words:
            previously_processed[prev_word] = 1
        
        def is_a_good_word(cur_word):

            if cur_word in self.guessed_words:
                return False

            for j in range(5):
                if self.green[j] != '' and self.green[j] != cur_word[j]:
                    return False

                if cur_word[j] in self.yellow[j]:
                    return False

                if cur_word[j] in self.wrong_letters:
                    return False
            
            in_green = set()
            for i in range(5):
                if self.green[i] != '':
                    in_green.add(self.green[i])
            
            for i in range(5):
                for x in self.yellow[i]:
                    if x not in in_green and x not in cur_word:
                        return False
            
            return True

        def add_good_word_score(cur_word):
            cur_score = 0
            for j in range(5):
                if self.green[j] == cur_word[j]:
                    cur_score -= 2
                
                elif cur_word[j] in self.correct_letters:
                    cur_score -= 1

            possible.append(cur_word)
            mp[cur_word] = [cur_score, self.word_scores[cur_word]]

        for cur_word in self.dictionary:                     
            if is_a_good_word(cur_word) == True:
                if not previously_processed[cur_word]:
                    add_good_word_score(cur_word)

        for prev_word in self.previous_best_words:
            if is_a_good_word(prev_word) == True:
                add_good_word_score(prev_word)


        possible.sort(key = lambda x : mp[x])
        self.previous_best_words = possible
        
        return possible

    def guess(self):
        if self.tries_left == 6:
            return 'tares'  
        
        elif self.tries_left in self.break_point:
            self.recalibrate()
            return self.get_first_word()
        
        possible = self.get_next_words(self.prev_guess)
        return possible[0]


    def startgame(self, answer = ''):
        self.reset_variables()
        
        if answer == '':
            print('Enter your word: ')
            answer = input()
        
        if self.full_ordered_dictionary[self.word_index[answer]] != answer:
            print("Invalid word entered : {} - | idx={} | d[idx]={}".format(answer, self.word_index[answer], self.full_ordered_dictionary[self.word_index[answer]]))
            print("Full dict: ",self.full_dictionary[:10])
            print("Full ord dict: ",self.full_ordered_dictionary[:10])
            print("dict: ",self.dictionary[:10])
            return
        
        self.answer = answer
        pred = ''
        count = 0
        
        while self.tries_left > 0:
            pred = self.guess()
            count += 1
            
            self.guessed_words.append(pred)
            
            self.display(pred, answer)
            self.tries_left -= 1
            
            if pred == answer or self.tries_left == 0:
                break
            
            self.prev_guess = pred
            
        if pred == answer:
            # print("SUCCESS!")
            return count
        else:
            # print("FAIL!")
            return -1
    
    
    



w = WordleSolver()
global_dict = w.full_dictionary

def automate(limit = len(global_dict)):
    passed = 0
    failed = 0
    
    total_tries = 0
    cnt_tried = 0
    
    success_rate = 0
    avg_tries = 0
        
    failed_words = []
    passed_points = [0 for i in range(6)]

    results = []

    prv = [0, 0]
    
    for i in range(limit):
        word = global_dict[i]
        # print("\nWord #{} - {} : ".format(i+1, word), end = ' ')
        verdict = w.startgame(word)
        if verdict == -1:
            # print("FAIL |",end = ' ')
            failed += 1
            failed_words.append(word)
        else:
            # print("Tries = {} | ".format(verdict),end = ' ')
            passed += 1
            cnt_tried += 1
            total_tries += verdict
            passed_points[verdict - 1] += 1
        # print(len(w.dictionary), len(w.full_dictionary), w.dictionary[0], w.full_dictionary[0], end=' ')
        
        if (i + 1)%100 == 0:
            success_rate = round(100 * (passed / (passed + failed)), 2)
            avg_tries = round(total_tries / cnt_tried, 3)

            print("#{} words: Average tries = {} | Success rate = {}% | Pass-Fail = {}/{}".format(i+1, avg_tries, success_rate, passed, failed))

            if (i+1)%1000 == 0:
                cur_success = round(100 * ( (passed - prv[0]) / (passed + failed - prv[0] - prv[1])), 3)

                cur = '#' + str(i+1) +" words: " + "avg_tries = " + str(avg_tries) + " | total_win = " + str(success_rate) + "% | "
                cur += "pass_fail = " + str(passed) + "/" + str(failed) + " | cur_win = " + str(cur_success) + "%"

                results.append(cur)

                prv = [passed, failed]
        
        # success_rate = round(100 * (passed / (passed + failed)), 2)
        # print("Success: {}%".format(success_rate))
    
    success_rate = round(100 * (passed / (passed + failed)), 2)
    avg_tries = round(total_tries / cnt_tried, 3)

    print("\n=============================================\n\n")
    for x in results:
        print(x)    
    print("\n=============================================")
    print("Results:")
    print("Average tries taken = {}".format(avg_tries))
    print("Success Rate = {}% - #Passed = {}, #Failed = {}\n".format(success_rate, passed, failed))
    print("Failed words = ", len(failed_words))
    print("Passed checkpoints: ", passed_points)

