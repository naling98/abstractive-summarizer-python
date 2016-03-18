import os
from gurobipy import *
import numpy as np
from model_loader import load_model
from scorer import Scorer
import linecache
import sys

VP_THRESHOLD = 0.75
MAX_WORD_LENGTH = 100
MAX_SENTENCE_NUMBER = 10
MIN_SENTENCE_LENGTH = 5
MIN_VERB_LENGTH = 2

PRONOUNS = ["it", "i", "you", "he", "they", "we", "she", "who", "them", "me", "him", "one", "her", "us", "something",
            "nothing", "anything", "himself", "everything", "someone", "themselves", "everyone", "itself", "anyone",
            "myself"]

# doc_ids = ["D30006T","D30037T","D30005T","D30049T","D30033T","D30001T","D30045T","D30053T",
#            "D31033T","D31001T","D31013T","D31009T","D30017T","D30029T","D30048T","D30024T",
#            "D30036T","D30044T","D30056T","D30020T","D30040T","D31032T","D31008T","D30008T",
#            "D30028T","D30015T","D30059T","D30027T","D30003T","D30047T","D30011T","D30055T",
#            "D30031T","D31031T","D30051T","D31043T","D30007T","D30026T","D30038T","D30002T",
#            "D30046T","D30022T","D31050T","D30034T","D30042T","D30010T","D31022T","D30050T",
#            "D31026T","D31038T"]

doc_ids = ["D30006T"]

if len(sys.argv) > 2:
    parent_directory = sys.argv[1] #this directory contains all folders of document sets
    doc_ids = os.listdir(parent_directory)
else:
    parent_directory = "/Users/chientran/Documents/jaist/abstractive_summarization/preprocessed_data/duc_2004"

def PrintException():
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    print 'EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj)

for doc_id in doc_ids:
    print "processing %s" % doc_id
    noun_phrases, verb_phrases, corefs, docs, indicator_matrix = \
        load_model("%s/%s" % (parent_directory, doc_id))

    def gamma_key(noun_id, verb_id):
        return "gamma_%d:%d" % (noun_id, verb_id)

    def phrase_to_phrase_key(phrase_1_id, phrase_2_id, is_np):
        if is_np:
            return "noun_%d:%d" % (phrase_1_id, phrase_2_id)
        else:
            return "verb_%d:%d" % (phrase_1_id, phrase_2_id)

    def score(phrases, docs):
        scores = []
        scorer = Scorer(docs)

        for id, phrase in phrases.items():
            p_score = scorer.calculate_score(phrase)
            phrase.score = p_score
            scores.append(p_score)

        scores.sort()

        return scores[len(scores)/2]

    def calculate_similarity(phrase_1, phrase_2):
        union_count = 0

        if len(phrase_1.concepts) == 0 or len(phrase_2.concepts) == 0:
            return 0.0

        for c in phrase_1.concepts:
            if c in phrase_2.concepts:
                union_count += 1

        return float(union_count) / (len(phrase_1.concepts) + len(phrase_2.concepts) - union_count)


    def build_alternative_np(noun_phrases, corefs):
        noun_ids = noun_phrases.keys()
        print len(noun_ids)
        alternative_matrix = np.zeros((max(noun_ids)+1, max(noun_ids)+1))

        for key, refs in corefs.items():
            cluster = set(refs)
            np_ids_in_same_cluster = []
            for id in noun_ids:
                noun = noun_phrases[id]
                if (noun.content in cluster):
                    np_ids_in_same_cluster.append(id)

            for idx, id in enumerate(np_ids_in_same_cluster):
                for id2 in np_ids_in_same_cluster[idx+1:]:
                    alternative_matrix[id, id2] = 1
                    alternative_matrix[id2, id] = 1

        return alternative_matrix


    def build_alternative_vp(verb_phrases):
        verb_ids = verb_phrases.keys()

        alternative_matrix = np.zeros((max(verb_ids) + 1, max(verb_ids) + 1))
        for i in xrange(len(verb_ids) - 1):
            for j in xrange(i+1, len(verb_ids)):
                verb_1 = verb_phrases[verb_ids[i]]
                verb_2 = verb_phrases[verb_ids[j]]

                similarity = calculate_similarity(verb_1, verb_2)

                if similarity >= VP_THRESHOLD:
                    alternative_matrix[i,j] = 1
                    alternative_matrix[j,i] = 1

        return alternative_matrix


    def build_compatibility_matrix(noun_phrases, verb_phrases, indicator_matrix, corefs):
        noun_ids = noun_phrases.keys()
        verb_ids = verb_phrases.keys()

        compatibility_matrix = np.zeros((max(noun_ids) + 1, max(verb_ids) + 1))

        print "Building alternative NP"
        alternative_np = build_alternative_np(noun_phrases, corefs)

        print "Building alternative VP"
        alternative_vp = build_alternative_vp(verb_phrases)

        alternative_vp_by_ids = {}

        for verb_id in verb_ids:
            alternative_vp_by_ids[verb_id] = [v_id for v_id in alternative_vp[verb_id, :] if alternative_vp[verb_id, v_id]]

        print "Building compatibility matrix"
        for idx1, noun_id in enumerate(noun_ids):
            alternative_np_ids = [n_id for n_id in alternative_np[noun_id, :] if alternative_np[noun_id, n_id]]
            for idx2, verb_id in enumerate(verb_ids):
                if indicator_matrix[noun_id, verb_id] == 1:
                    compatibility_matrix[noun_id, verb_id] = 1
                    continue

                flag = False

                #print alternative_np_ids

                for other_noun_id in alternative_np_ids:
                    if indicator_matrix[other_noun_id, verb_id] == 1:
                        compatibility_matrix[noun_id, verb_id] = 1
                        flag = True
                        break

                if flag:
                    continue

                #alternative_vp_ids = [v_id for v_id in alternative_vp[verb_id, :] if alternative_vp[verb_id, v_id]]
                #print alternative_vp_ids

                for other_verb_id in alternative_vp_by_ids[verb_id]:
                    if indicator_matrix[noun_id, other_verb_id] == 1:
                        compatibility_matrix[noun_id, verb_id] = 1
                        break

        return compatibility_matrix, alternative_np, alternative_vp


    def add_np_validity_constraint(model):
        for n_id, noun_phrase in noun_phrases.items():
            noun_var = noun_vars[n_id]
            noun_constr = LinExpr()
            noun_constr += -noun_var
            for v_id, verb_phrase in verb_phrases.items():
                if compatibility_matrix[n_id, v_id] == 1:
                    gamma_var = gamma_vars.get(gamma_key(n_id, v_id))
                    noun_constr += gamma_var
                    model.addConstr(noun_var - gamma_var >= 0, "np_validity_%s" % gamma_key(n_id, v_id))

            model.addConstr(noun_constr, GRB.GREATER_EQUAL, 0.0, "np_validity_%d" % n_id)

    def add_vp_validity_constraint(model):
        for v_id, verb_phrase in verb_phrases.items():
            verb_var = verb_vars[v_id]
            verb_constr = LinExpr()
            verb_constr += -verb_var

            for n_id, noun_phrase in noun_phrases.items():
                if compatibility_matrix[n_id, v_id] == 1:
                    gamma_var = gamma_vars.get(gamma_key(n_id, v_id))
                    verb_constr += gamma_var

            model.addConstr(verb_constr, GRB.EQUAL, 0.0, "vp_validity_%d" % v_id)


    def add_not_i_within_i_constraint(model, phrases, vars):
        phrase_ids = phrases.keys()
        for idx, i in enumerate(phrase_ids):
            for j in phrase_ids[idx+1:]:
                phrase_1 = phrases[i]
                phrase_2 = phrases[j]

                if phrase_1.id == phrase_2.parentId:
                    var_1 = vars[i]
                    var_2 = vars[j]
                    model.addConstr(var_1 + var_2 <= 1.0, "i_within_i_%s_%d_%d" % (phrase_1.is_np, i, j))


    def add_phrase_coocurrence_constraint(model, phrases, vars, cooccurence_vars):
        phrase_ids = phrases.keys()
        for idx, i in enumerate(phrase_ids):
            phrase_1 = phrases[i]
            var_1 = vars[i]
            for j in phrase_ids[idx+1:]:
                var_2 = vars[j]
                coocurrence_var = cooccurence_vars["%d:%d" % (i,j)]

                model.addConstr(coocurrence_var - var_2 <= 0, "p_occ_1_%s_%d_%d" % (phrase_1.is_np, i, j))
                model.addConstr(coocurrence_var - var_1 <= 0, "p_occ_2_%s_%d_%d" % (phrase_1.is_np, i, j))
                model.addConstr(var_1 + var_2 - coocurrence_var <= 1.0, "p_occ_3_%s_%d_%d" % (phrase_1.is_np, i, j))


    def add_sentence_number_constraint(model):
        exp = LinExpr()
        for np_id in noun_phrases.keys():
            var = noun_vars[np_id]
            exp += var

        model.addConstr(exp, GRB.LESS_EQUAL, MAX_SENTENCE_NUMBER, "sentence_number")


    def add_short_sentence_avoidance_constraint(model):
        for vp_id, phrase in verb_phrases.items():
            if phrase.sentence_length < MIN_SENTENCE_LENGTH or phrase.word_length < MIN_VERB_LENGTH:
                var = verb_vars[vp_id]
                model.addConstr(var == 0, "short_sentence_avoidance %d" % vp_id)


    def add_pronoun_avoidance_constraint(model):
        for np_id, phrase in noun_phrases.items():
            if phrase.content in PRONOUNS:
                var = noun_vars[np_id]
                model.addConstr(var == 0, "pronoun_avoidance %d" % np_id)


    def add_length_constraint(model):
        exp = LinExpr()
        for np_id, phrase in noun_phrases.items():
            var = noun_vars[np_id]
            exp += phrase.word_length*var

        for vp_id, phrase in verb_phrases.items():
            var = verb_vars[vp_id]
            exp += phrase.word_length*var

        model.addConstr(exp, GRB.LESS_EQUAL, MAX_WORD_LENGTH, "length_constraint")

    def add_low_score_avoidance_constraint(model):
        for np_id, phrase in noun_phrases.items():
            if phrase.score < mean_noun_score*0.3:
                var = noun_vars[np_id]
                model.addConstr(var == 0, "low_score_np_%d" % np_id)

        for vp_id, phrase in verb_phrases.items():
            if phrase.score < mean_verb_score*0.3:
                var = verb_vars[vp_id]
                model.addConstr(var == 0, "low_score_vp_%d" % np_id)


    total_noun_score = score(noun_phrases, docs)
    total_verb_score = score(verb_phrases, docs)

    mean_noun_score = total_noun_score #/ len(noun_phrases.keys())
    mean_verb_score = total_verb_score #/ len(verb_phrases.keys())

    compatibility_matrix, alternative_np, alternative_vp = build_compatibility_matrix(noun_phrases, verb_phrases, indicator_matrix, corefs)

    print "Building optimization model"
    noun_vars = {}
    noun_to_noun_vars = {}
    verb_vars = {}
    verb_to_verb_vars = {}
    gamma_vars = {}

    try:
        model = Model("mip")
        noun_ids = noun_phrases.keys()
        verb_ids = verb_phrases.keys()

        obj = LinExpr()

        for index, n_id in enumerate(noun_ids):
            noun_vars[n_id] = model.addVar(vtype=GRB.BINARY, name="noun_%d" % n_id)
            noun_1 = noun_phrases[n_id]
            obj.addTerms(noun_1.score, noun_vars[n_id])

            for other_n_id in noun_ids[index+1:]:
                var = model.addVar(vtype=GRB.BINARY, name="noun_%d:%d" % (n_id, other_n_id))

                noun_2 = noun_phrases[other_n_id]

                score = -(noun_1.score + noun_2.score)

                if alternative_np[n_id,other_n_id] != 1:
                    score *= calculate_similarity(noun_1, noun_2)

                obj.addTerms(score, var)
                noun_to_noun_vars["%d:%d" % (n_id, other_n_id)] = var

            for v_id in verb_ids:
                if compatibility_matrix[n_id, v_id] == 1:
                    gamma_vars[gamma_key(n_id, v_id)] = model.addVar(vtype=GRB.BINARY, name="gamma_%d:%d" % (n_id, v_id))

        for index, v_id in enumerate(verb_ids):
            verb_vars[v_id] = model.addVar(vtype=GRB.BINARY, name="verb_%d" % v_id)
            obj.addTerms(verb_phrases[v_id].score, verb_vars[v_id])
            for other_v_id in verb_ids[index+1:]:
                var = model.addVar(vtype=GRB.BINARY, name="verb_%d:%d" % (v_id, other_v_id))
                score = -(verb_phrases[v_id].score + verb_phrases[other_v_id].score) * \
                        calculate_similarity(verb_phrases[v_id], verb_phrases[other_v_id])
                obj.addTerms(score, var)
                verb_to_verb_vars["%d:%d" % (v_id, other_v_id)] = var

        model.update()
        model.setObjective(obj, GRB.MAXIMIZE)

        print "Adding constraint"
        add_np_validity_constraint(model)
        add_vp_validity_constraint(model)
        add_not_i_within_i_constraint(model, noun_phrases, noun_vars)
        add_not_i_within_i_constraint(model, verb_phrases, verb_vars)
        add_phrase_coocurrence_constraint(model, noun_phrases, noun_vars, noun_to_noun_vars)
        add_phrase_coocurrence_constraint(model, verb_phrases, verb_vars, verb_to_verb_vars)
        add_short_sentence_avoidance_constraint(model)
        add_sentence_number_constraint(model)
        add_pronoun_avoidance_constraint(model)
        add_length_constraint(model)
        add_low_score_avoidance_constraint(model)

        model.optimize()

        sentences = {}
        for key, var in gamma_vars.items():
            if var.x == 1:
                np_id, vp_id = key.split(":")
                np_id = int(np_id.split("_")[1])
                vp_id = int(vp_id)

                if not sentences.has_key(np_id):
                    sentences[np_id] = [vp_id]
                else:
                    sentences[np_id].append(vp_id)

        sentences_in_ordered = {}

        for np_id, vp_ids in sentences.items():
            min_vp_id = min(vp_ids)
            sentences_in_ordered[min_vp_id] = np_id

        keys_ordered = sentences_in_ordered.keys()
        keys_ordered.sort()

        print keys_ordered

        summary = []

        for id in keys_ordered:
            np_id = sentences_in_ordered[id]
            s = [noun_phrases[np_id].content]
            for v_id in sentences[np_id]:
                s.append(verb_phrases[v_id].content)

            summary.append(' '.join(s))

        print '\n'.join(summary)

        with open("output/%s_system.txt" % doc_id, "w") as f:
            f.write('\n'.join(summary))

    except GurobiError as e:
        print 'Error reported'
        print e
        PrintException()
