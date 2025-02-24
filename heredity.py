import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}

PARENTS = [0.01, 0.5, 0.99]


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    result = 1
    # details = [] - used details to validate joint_probability
    
    for person in people:

        number_of_genes = 0
        if person in one_gene:
            number_of_genes = 1
        elif person in two_genes:
            number_of_genes = 2
            
        gene_prob = gene_probability(number_of_genes, person, people, one_gene, two_genes)

        has_treat = False
        if person in have_trait:
            has_treat = True

        # has_trait = person in have_trait
                
        trait_chance = PROBS.get("trait")[number_of_genes][has_treat]
        result = result * gene_prob * trait_chance
        
        # details.append((person, number_of_genes, has_trait))

    # print(f"{details} : {result:.8f}")
    return result


def gene_probability(number_of_genes, person, people, one_gene, two_genes):
    person_info = people.get(person)
    mother = person_info.get('mother')
    father = person_info.get('father')
    gene_prob = 0

    # suppose person has 2 parents or no parents at all. We don't cover the case with 1 parent. 
    if not mother and not father:
        gene_prob = PROBS.get("gene")[number_of_genes]
    elif mother and father:
        mother_genes = 1 if mother in one_gene else 2 if mother in two_genes else 0
        father_genes = 1 if father in one_gene else 2 if father in two_genes else 0
        
        from_mother = PARENTS[mother_genes]
        from_father = PARENTS[father_genes]
        not_from_mother = (1 - from_mother)
        not_from_father = (1 - from_father)
        
        # gene_probability related to the number_of_genes
        if number_of_genes == 0:
            gene_prob = not_from_mother * not_from_father
        elif number_of_genes == 1:
            gene_prob = from_mother*not_from_father + from_father*not_from_mother
        else:
            gene_prob = from_mother * from_father

    # print(f"Person: {person}, Number of Genes: {number_of_genes}, Gene Probability: {gene_prob:.8f}")
    return gene_prob


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    for person in probabilities:
        person_genes = 1 if person in one_gene else 2 if person in two_genes else 0
        person_trait = 1 if person in have_trait else 0

        probabilities[person]["gene"][person_genes] = probabilities[person]["gene"][person_genes] + p
        probabilities[person]["trait"][person_trait] = probabilities[person]["trait"][person_trait] + p


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    for person in probabilities:
        sum = 0
        for gene_prob in range(3): 
            sum = sum + probabilities[person]["gene"][gene_prob]
        k = 1/sum
        for gene_prob in range(3): 
            probabilities[person]["gene"][gene_prob] = probabilities[person]["gene"][gene_prob] * k

        sum = 0
        for trait_prob in range(2): 
            sum = sum + probabilities[person]["trait"][trait_prob]
        k = 1/sum
        for trait_prob in range(2): 
            probabilities[person]["trait"][trait_prob] = probabilities[person]["trait"][trait_prob] * k


if __name__ == "__main__":
    main()
