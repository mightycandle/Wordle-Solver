
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
        self.n_grams_scores = self.make_n_grams()
        
        self.full_position_count = self.make_position_count()
        self.position_count = self.full_position_count.copy()        
        
        self.init_frequent_letters, self.init_letter_rank, self.init_letter_best_index = self.get_alphabet_stats()
        self.frequent_letters = self.init_frequent_letters.copy()
        self.letter_rank = self.init_letter_rank.copy()
        self.letter_best_index = self.init_letter_best_index.copy()
        
        self.init_word_scores = self.get_word_scores()
        self.word_scores = self.init_word_scores.copy()
        
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
        
        # best starting guess the model predicted -> from the function get_start_word()
        self.best_start_word = 'tares'
        
    
    def reset_variables(self):
        if len(self.dictionary) != len(self.full_dictionary):
            self.dictionary = self.full_ordered_dictionary.copy()
            
            self.position_count = self.full_position_count.copy()
            
            self.frequent_letters = self.init_frequent_letters.copy()
            self.letter_rank = self.init_letter_rank.copy()
            self.letter_best_index = self.init_letter_best_index.copy()
            
            self.word_scores = self.init_word_scores.copy()
           
        self.tries_left = 6
        
        self.guessed_words = []
        self.correct_letters = set()
        self.wrong_letters = set()
        
        self.answer = ''
        self.green = ['' for i in range(5)]
        self.yellow = [[] for i in range(5)]        
        
        self.prev_guess = ''
        self.previous_best_words = []
            
    def recalibrate(self):
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
                    
            return new_dict
        
        self.dictionary = update_dictionary()
        
        self.position_count = self.get_position_count()
        
        self.frequent_letters = self.get_frequent_letters()
        self.letter_rank = self.get_letter_ranks()
        self.letter_best_index = self.get_letter_best_index()
        
        self.word_scores = self.get_word_scores()
        
    def build_dictionary(self):
        text_file = open(self.dictionary_file_location,"r")
        full_dictionary = text_file.read().splitlines()
        text_file.close()
        return full_dictionary
    
    def make_n_grams(self):
        
        def hash_value(s):
            p = 31
            hash = 0
            for i in range(3):
                hash += (1 + (ord(s[i]) - ord('a')))*(p ** i)
            return hash
        
        mp = [[0 for i in range(3)] for i in range(26800)]
        
        for word in self.full_dictionary:
            for i in range(3):
                mp[hash_value(word[i:i + 3])][i] += 1
        
        n_gram_scores = []
        for word in self.full_dictionary:
            cur_score = 1.0
            for i in range(3):
                h = hash_value(word[i:i + 3])
                cur_score *= (mp[h][i] / sum(mp[h]))
            n_gram_scores.append(cur_score)
            
        n_gram_scores = zscore(n_gram_scores)

        mp = dd(float)
        for i, j in enumerate(self.full_dictionary):
            mp[j] = n_gram_scores[i]
            
        return mp
    
    def make_position_count(self):
        position_count = [[0 for i in range(5)] for i in range(26)]
        
        for word in self.dictionary:
            for i in range(5):
                position_count[ord(word[i]) - ord('a')][i] += 1

        return position_count              
    
    def get_alphabet_stats(self):
        frequency = dd(int)
        for word in self.dictionary:
            for letter in set(word):
                frequency[letter] += 1

        letters = self.alphabets.copy()
        letters.sort(key = lambda x: -frequency[x])
        
        letter_ranks = dd(str)
        for i in range(26):
            letter_ranks[self.frequent_letters[i]] = i + 1
        
        best_position = dd(list)
        
        for letter in self.alphabets:
            freq = self.position_count[ord(letter) - ord('a')]
            track = []
            for i in range(5):
                track.append([-freq[i], i])
            track.sort()
            
            for i in range(2):
                best_position[letter].append(track[i][1])
        
        return letters, letter_ranks, best_position
           
    def get_word_scores(self):
        mp = dd(float)
        for word in self.dictionary:
            mp[word] = self.get_score(word)
        
        if len(self.dictionary) == len(self.full_ordered_dictionary):
            if self.dictionary[0] != self.best_start_word:
                self.dictionary.sort(key = lambda x: mp[x])
                
        else:
            self.dictionary.sort(key = lambda x: mp[x])
        
        if self.full_ordered_dictionary[0] != self.best_start_word:
            self.full_ordered_dictionary.sort(key = lambda x: mp[x])
        
        return mp
    
    def get_score(self, word):
        cur_score = 0.0
        visited = [0 for i in range(26)]

        for i in range(5):
            letter = word[i]
            if visited[ord(letter) - ord('a')]:
                cur_score += 5 * self.letter_rank[letter]
            else:
                if i in self.letter_best_index[letter]:
                    cur_score += 1 * self.letter_rank[letter]
                else:
                    cur_score += 5 * self.letter_rank[letter]
            visited[ord(letter) - ord('a')] = 1
        
        unique_count = len(set(word))
        cur_score += (5 - unique_count) * 13

        return cur_score - (unique_count - 3.5)*self.n_grams_scores[word]
         
    def get_start_word(self):
        
        def result_of_guess(answer, guess):
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
        
        good_words = self.full_dictionary
        good_words.sort(key = lambda x: self.word_scores[x])
        
        mp = dd(int)
        
        for answer in good_words[:500]:
            cur = 0
            for word in self.dictionary:
                cur_stats = result_of_guess(answer, word)
                
                cur += cur_stats[0] * (3)
                cur += cur_stats[1] * (2)
                cur += cur_stats[2] * (-1)
            
            mp[answer] = cur
        
        good_words.sort(key = lambda x: -mp[x])
        
        return good_words[0]
        
    def evaluate_guess(self, prediction, answer):
        self.guessed_words.append(prediction)
        result = [0 for i in range(5)]
        
        for i, letter in enumerate(prediction):
            if letter not in answer:
                result[i] = 0
                self.wrong_letters.add(letter)            
            else:
                self.correct_letters.add(letter)
                
                if letter != answer[i]:
                    result[i] = 1
                    self.yellow[i].append(letter)
                else:
                    result[i] = 2
                    self.green[i] = letter
        
        return result
               
    def get_next_word(self, word):
        if self.tries_left == 6:
            return self.best_start_word  
        
        elif self.tries_left in self.break_point:
            self.recalibrate()
            
            return self.get_start_word()
    
        possible_words = []
        mp = dd(list)

        is_prev_best = dd(int)
        for w in self.previous_best_words:
            is_prev_best[w] = 1
        
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

            possible_words.append(cur_word)
            mp[cur_word] = [cur_score, self.word_scores[cur_word]]

        for w in self.dictionary:                     
            if is_a_good_word(w) == True:
                if not is_prev_best[w]:
                    add_good_word_score(w)

        for w in self.previous_best_words:
            if is_a_good_word(w) == True:
                add_good_word_score(w)

        possible_words.sort(key = lambda x : mp[x])
        self.previous_best_words = possible_words
        
        return possible_words[0]
        
    def start_game(self, answer = '', multiple_runs = False):
        self.reset_variables()
        
        if answer == '':
            print('Enter your word: ')
            answer = input()
        
        if self.word_scores[answer] == 0:
            print("Invalid word")
            return
        
        self.answer = answer
        prediction = ''
        
        while self.tries_left > 0:
            prediction = self.get_next_word(self.prev_guess)
            
            result = self.evaluate_guess(prediction, answer)
            
            if multiple_runs == False:
                print("\nGuess #{} - {}".format(7 - self.tries_left, prediction))
                print(result)
            
            self.tries_left -= 1
            
            if prediction == answer or self.tries_left == 0:
                break
            
            self.prev_guess = prediction
            
        if prediction == answer:
            if multiple_runs == False:
                print("\nSUCCESS!")
                
            return 6 - self.tries_left
        else:
            if multiple_runs == False:
                print("\nFAIL!")
            return -1
    
    
        
w = WordleSolver()
all_words = w.full_dictionary

def automate(limit = len(all_words), period = 1000):
    pass_fail = [0, 0]
    
    total_tries = 0
    cnt_tried = 0
    
    passed_points = [0 for i in range(6)]

    for i in range(limit):
        word = all_words[i]
        verdict = w.start_game(word,True)
        if verdict == -1:
            pass_fail[1] += 1
        else:
            pass_fail[0] += 1
            cnt_tried += 1
            total_tries += verdict
            passed_points[verdict - 1] += 1
        
        if (i + 1)%period == 0:
            success_rate = round(100 * (pass_fail[0] / sum(pass_fail)), 2)
            avg_tries = round(total_tries / cnt_tried, 3)
            
            print("#{} words: Average tries = {} | Success rate = {}% | Pass-Fail = {}/{}".format(i+1, avg_tries, success_rate, pass_fail[0], pass_fail[1]))

    
    success_rate = round(100 * (pass_fail[0] / sum(pass_fail)), 2)
    avg_tries = round(total_tries / cnt_tried, 3)
    
    print("\n=============================================")
    print("Results:")
    print("Average tries taken = {}".format(avg_tries))
    print("Success Rate = {}% - #Passed = {}, #Failed = {}\n".format(success_rate, pass_fail[0], pass_fail[1]))
    print("Passed checkpoints: ", passed_points)

