#include <math.h>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <random>
#include <stack>
#include <vector>

#include "rocketCentering.h"

using namespace std;

// return a double unifomrly sampled in (0,1)
double randDouble(mt19937& rng) {
    return uniform_real_distribution<>{0, 1}(rng);
}

// return uniformly sampled 0 or 1
bool randChoice(mt19937& rng) {
    return uniform_int_distribution<>{0, 1}(rng);
}

// return a random integer uniformly sampled in (min, max)
int randInt(mt19937& rng, const int& min, const int& max) {
    return uniform_int_distribution<>{min, max}(rng);
}

// return true if op is a suported operation, otherwise return false
bool isOp(string op) {
    if (op == "+")
        return true;
    else if (op == "-")
        return true;
    else if (op == "*")
        return true;
    else if (op == "/")
        return true;
    else if (op == ">")
        return true;
    else if (op == "abs")
        return true;
    else
        return false;
}

int arity(string op) {
    if (op == "abs")
        return 1;
    else
        return 2;
}

typedef string Elem;

// Fetch any operator
Elem fetchOperator(mt19937& rng) {
    static const vector<Elem> operators = {"+", "-", "*", "/", ">", "abs"};
    uniform_int_distribution<int> uniform_dist(0, operators.size() - 1);
    return operators[uniform_dist(rng)];
}

// Fetch any operand
Elem fetchOperand(mt19937& rng) {
    uniform_real_distribution<double> uniform_dist(-10, 10);
    double rand_double = uniform_dist(rng);
    vector<Elem> operands = {"a", "b", to_string(rand_double)}; // a is the rocket position, b is the rocket velocity. 1/3 chance of being a constant
    uniform_int_distribution<int> index(0, operands.size() - 1);
    return operands[index(rng)];
}

// Fetch any operator or operand
Elem fetchAny(mt19937& rng) {
    return randChoice(rng) ? fetchOperator(rng) : fetchOperand(rng);
}

class LinkedBinaryTree {
    public:
    struct Node {
        Elem elt;
        string name;
        Node* par;
        Node* left;
        Node* right;
        Node() : elt(), par(NULL), name(""), left(NULL), right(NULL) {}
        int depth() {
            if (par == NULL) return 0;
            return par->depth() + 1;
        }
    };

    class Position {
        private:
            Node* v;

        public:
            Position(Node* _v = NULL) : v(_v) {}
            Elem& operator*() { return v->elt; }
            Position left() const { return Position(v->left); }
            void setLeft(Node* n) { v->left = n; }
            Position right() const { return Position(v->right); }
            void setRight(Node* n) { v->right = n; }
            Position parent() const  // get parent
            {
            return Position(v->par);
            }
            bool isRoot() const  // root of the tree?
            {
            return v->par == NULL;
            }
            bool isExternal() const  // an external node?
            {
            return v->left == NULL && v->right == NULL;
            }
            friend class LinkedBinaryTree;  // give tree access
    };
    typedef vector<Position> PositionList;

    public:
    LinkedBinaryTree() : _root(NULL), score(0), steps(0), generation(0) {}

    // copy constructor
    LinkedBinaryTree(const LinkedBinaryTree& t) {
        _root = copyPreOrder(t.root());
        score = t.getScore();
        steps = t.getSteps();
        generation = t.getGeneration();
    }

    // copy assignment operator
    LinkedBinaryTree& operator=(const LinkedBinaryTree& t) {
        if (this != &t) {
            // if tree already contains data, delete it
            if (_root != NULL) {
                PositionList pl = positions();
                for (auto& p : pl) delete p.v;
            }
            _root = copyPreOrder(t.root());
            score = t.getScore();
            steps = t.getSteps();
            generation = t.getGeneration();
        }
        return *this;
    }

    // destructor
    ~LinkedBinaryTree() {
        if (_root != NULL) {
            PositionList pl = positions();
            for (auto& p : pl) delete p.v;
        }
    }

    int size() const { return size(_root); }
    int size(Node* root) const;
    int depth() const;
    bool empty() const { return size() == 0; };
    Node* root() const { return _root; }
    PositionList positions() const;
    void addRoot() { _root = new Node; }
    void addRoot(Elem e) {
        _root = new Node;
        _root->elt = e;
    }
    void nameRoot(string name) { _root->name = name; }
    void addLeftChild(const Position& p, const Node* n);
    void addLeftChild(const Position& p);
    void addRightChild(const Position& p, const Node* n);
    void addRightChild(const Position& p);
    void printExpression() { printExpression(_root); }
    void printExpression(Node* v);
    double evaluateExpression(double a, double b) {
        return evaluateExpression(Position(_root), a, b);
    };
    double evaluateExpression(const Position& p, double a, double b);
    long getGeneration() const { return generation; }
    void setGeneration(int g) { generation = g; }
    double getScore() const { return score; }
    void setScore(double s) { score = s; }
    double getSteps() const { return steps; }
    void setSteps(double s) { steps = s; }
    void randomExpressionTree(Node* v, const int& max_depth, mt19937& rng);
    void randomExpressionTree(const int& max_depth, mt19937& rng) {
        randomExpressionTree(_root, max_depth, rng);
    }
    void generateOperationNodes(Node* node, const int& max_depth, mt19937& rng);
    void addOperandNodes(Node* node, mt19937& rng);
    void deleteSubtreeMutator(mt19937& rng);
    void addSubtreeMutator(mt19937& rng, const int max_depth);
    void crossover(LinkedBinaryTree& otherTree, mt19937& rng);
    void deallocTree(Node* node);

    protected:                                        // local utilities
        void preorder(Node* v, PositionList& pl) const;  // preorder utility
        void findValidPositions(Node* node, PositionList& possible_nodes);
        Node* copyPreOrder(const Node* root);
        double score;     // mean reward over 20 episodes
        double steps;     // mean steps-per-episode over 20 episodes
        long generation;  // which generation was tree "born"
    private:
        Node* _root;  // pointer to the root
};

// add the tree rooted at node child as this tree's left child
void LinkedBinaryTree::addLeftChild(const Position& p, const Node* child) {
    Node* v = p.v;
    v->left = copyPreOrder(child);  // deep copy child
    v->left->par = v;
}

// Add the tree rooted at node child as this tree's right child
void LinkedBinaryTree::addRightChild(const Position& p, const Node* child) {
    Node* v = p.v;
    v->right = copyPreOrder(child);  // deep copy child
    v->right->par = v;
}

void LinkedBinaryTree::addLeftChild(const Position& p) {
    Node* v = p.v;
    v->left = new Node;
    v->left->par = v;
}

void LinkedBinaryTree::addRightChild(const Position& p) {
    Node* v = p.v;
    v->right = new Node;
    v->right->par = v;
}

// Return a list of all nodes
LinkedBinaryTree::PositionList LinkedBinaryTree::positions() const {
    PositionList pl;
    preorder(_root, pl);
    return PositionList(pl);
}

void LinkedBinaryTree::preorder(Node* v, PositionList& pl) const {
    pl.push_back(Position(v));
    if (v->left != NULL) preorder(v->left, pl);
    if (v->right != NULL) preorder(v->right, pl);
}

int LinkedBinaryTree::size(Node* v) const {
    if (v == NULL) return 0;
    int lsize = 0;
    int rsize = 0;
    if (v->left != NULL) lsize = size(v->left);
    if (v->right != NULL) rsize = size(v->right);
    return 1 + lsize + rsize;
}

int LinkedBinaryTree::depth() const {
    PositionList pl = positions();
    int depth = 0;
    for (auto& p : pl) depth = max(depth, p.v->depth());
    return depth;
}

LinkedBinaryTree::Node* LinkedBinaryTree::copyPreOrder(const Node* root) {
    if (root == NULL) return NULL;
    Node* nn = new Node;
    nn->elt = root->elt;
    nn->left = copyPreOrder(root->left);
    if (nn->left != NULL) nn->left->par = nn;
    nn->right = copyPreOrder(root->right);
    if (nn->right != NULL) nn->right->par = nn;
    return nn;
}

void LinkedBinaryTree::printExpression(Node* v) {
    // Base case; if v's parent is leaf
    if (v == nullptr) {
        return;
    }

    // Determine the type of node
    vector<string> operators = {"+", "-", "*", "/", ">", "abs"};
    bool is_operator = find(operators.begin(), operators.end(), v->elt) != operators.end();
    bool has_arity_1 = arity(v->elt) == 1;

    // Recur on children inorder
    if (is_operator && !has_arity_1) {
        // Operator with arity 2
        cout << "(";
        printExpression(v->left);
        cout << v->elt;
        printExpression(v->right);
        cout << ")";
    } else if (has_arity_1) {
        // Operator with arity 1
        cout << v->elt << "(";
        printExpression(v->left);
        cout << ")";
    } else {
        // Operand
        cout << v->elt;
    }
}

double evalOp(string op, double x, double y = 0) {
    double result;
    if (op == "+")
        result = x + y;
    else if (op == "-")
        result = x - y;
    else if (op == "*")
        result = x * y;
    else if (op == "/") {
        result = x / y;
    } else if (op == ">") {
        result = x > y ? 1 : -1;
    } else if (op == "abs") {
        result = abs(x);
    } else
        result = 0;
    return isnan(result) || !isfinite(result) ? 0 : result;
}

double LinkedBinaryTree::evaluateExpression(const Position& p, double a, double b) {
    if (!p.isExternal()) {
        auto x = evaluateExpression(p.left(), a, b);
        if (arity(p.v->elt) > 1) {
            auto y = evaluateExpression(p.right(), a, b);
            return evalOp(p.v->elt, x, y);
        } else {
            return evalOp(p.v->elt, x);
        }
    } else {
        if (p.v->elt == "a")
            return a;
        else if (p.v->elt == "b")
            return b;
        else
            return stod(p.v->elt);
    }
}

// Pick random position and delete the subtree rooting from there
void LinkedBinaryTree::deleteSubtreeMutator(mt19937& rng) {
    // Select a random position
    if (_root == nullptr) return;
    PositionList positions = this->positions();
    if (positions.empty()) return;
    Position random_position = positions[randInt(rng, 0, positions.size() - 1)];

    // Deallocate the subtree rooted at the random position
    deallocTree(random_position.v);
}

// Recursively generate a random expression tree
void LinkedBinaryTree::randomExpressionTree(Node* v, const int& max_depth, mt19937& rng) {
    if (isOp(v->elt)) {
        if (v->depth() >= max_depth - 1) {
            addOperandNodes(v, rng);
        } else {
            generateOperationNodes(v, max_depth, rng);
        }
    }
}

// Generate operation nodes for the tree
void LinkedBinaryTree::generateOperationNodes(Node* node, const int& max_depth, mt19937& rng) {
    // Add left child with random operator
    unique_ptr<Node> left_child(new Node);
    left_child->elt = fetchAny(rng);
    addLeftChild(node, left_child.get());

    // Recur on left child if it is an operation
    if (isOp(node->left->elt)) {
        randomExpressionTree(node->left, max_depth, rng);
    }

    // Add right child if the operation has arity > 1
    if (arity(node->elt) > 1) {
        unique_ptr<Node> right_child(new Node);
        right_child->elt = fetchAny(rng);
        addRightChild(node, right_child.get());

        if (isOp(node->right->elt)) {
            randomExpressionTree(node->right, max_depth, rng);
        }
    }
}

// Add operand nodes to the tree
void LinkedBinaryTree::addOperandNodes(Node* node, mt19937& rng) {
    Node* left_operand = new Node;
    left_operand->elt = fetchOperand(rng);
    addLeftChild(node, left_operand);
    delete left_operand;

    // Add right operand if the operation has arity > 1
    if (arity(node->elt) > 1) {
        Node* right_operand = new Node;
        right_operand->elt = fetchOperand(rng);
        addRightChild(node, right_operand);
        delete right_operand;
    }
}

// Add a randomly generated subtree to the tree at any node where an operation can be added
void LinkedBinaryTree::addSubtreeMutator(mt19937& rng, const int max_depth) {
    if (empty()) {
        // If the tree is empty, then the created tree is the new tree with a random operator as the root
        addRoot(fetchOperator(rng));
        randomExpressionTree(max_depth, rng);
    } else {
        // Recursively et list of possible nodes, then choose one
        PositionList possible_nodes;
        findValidPositions(_root, possible_nodes);
        Position parent = possible_nodes[randInt(rng, 0, possible_nodes.size() - 1)];

        // Add a subtree to the selected position with random operator and random expression tree
        LinkedBinaryTree subtree;
        subtree.addRoot(fetchAny(rng));
        subtree.randomExpressionTree(max_depth - parent.v->depth() - 1, rng);

        // Connect subtree to parent tree
        if (parent.v->left == nullptr) {
            addLeftChild(parent, subtree.root());
        } else {
            addRightChild(parent, subtree.root());
        }
    }
}

// Find all positions in the tree where an operation can be added
void LinkedBinaryTree::findValidPositions(Node* node, PositionList& possible_nodes) {
    if (node == nullptr) return;

    // If the node is an operation and has a children < arity (expected more arguments)
    if (isOp(node->elt) && (node->left == nullptr || (arity(node->elt) == 2 && node->right == nullptr))) {
        possible_nodes.push_back(Position(node));
    }

    // Recursive on its children in preorder fashion
    findValidPositions(node->left, possible_nodes);
    findValidPositions(node->right, possible_nodes);
}

// Overloaded < operator for sorting trees
bool operator<(const LinkedBinaryTree& x, const LinkedBinaryTree& y) {
    return x.getScore() < y.getScore();
}

// Comparator struct for sorting trees
struct LexLessThan {
    bool operator()(const LinkedBinaryTree& TA, const LinkedBinaryTree& TB) const {
        if (abs(TA.getScore() - TB.getScore()) < 0.01) {
            return TA.size() > TB.size();
        } else {
            return TA.getScore() < TB.getScore();
        }
    }
};

// Create an expression tree from a postfix expression
LinkedBinaryTree createExpressionTree(string postfix) {
    stack<LinkedBinaryTree> tree_stack;
    stringstream ss(postfix);

    // Split each line into words
    string token;
    while (getline(ss, token, ' ')) {
        LinkedBinaryTree t;
        if (!isOp(token)) {
            t.addRoot(token);
            tree_stack.push(t);
        } else {
            t.addRoot(token);
            if (arity(token) > 1) {
                LinkedBinaryTree r = tree_stack.top();
                tree_stack.pop();
                t.addRightChild(t.root(), r.root());
            }
            LinkedBinaryTree l = tree_stack.top();
            tree_stack.pop();
            t.addLeftChild(t.root(), l.root());
            tree_stack.push(t);
        }
    }
    return tree_stack.top();
}

// Create a random expression tree with a maximum depth
LinkedBinaryTree createRandExpressionTree(int max_depth, mt19937& rng) {
    vector<string> operators = {"+", "-", "*", "/", ">", "abs"};
    vector<string> operands = {"a", "b"};
    string postfix;
    
    // Select depth for current tree as a rand int e [1, max_depth]
    int depth = randInt(rng, 1, max_depth);
    int total_arity = 1; // min arity is 1

    // Generate stack of random operators with length equal to depth
    stack<string> operator_sequence;
    for (int i = 0; i < depth; i++) {
        int rand_op_index = randInt(rng, 0, operators.size()-1);
        string op = operators[rand_op_index];
        operator_sequence.push(op);
        total_arity += arity(op)-1;
    }
    int operands_to_use = total_arity; // need variable to remain constant, while another variable decrements
    
    // Construct string of postfix expression
    for (int i = 0; i < total_arity+depth; i++) {
        if (i > 0) {
            // Add space after each operator and operand
            postfix += " ";
        }
        if ((i == max_depth-1 || randChoice(rng) || i < arity(operator_sequence.top()) || operator_sequence.size() == 0 || (depth-operator_sequence.size())>=(total_arity-operands_to_use)-1) && operands_to_use > 0) {
            // Add operand
            postfix += operands[randInt(rng, 0, operands.size() - 1)];
            operands_to_use--;
        } else { 
            // Add operator
            postfix += operator_sequence.top();
            operator_sequence.pop();
        }
    }

    // cout << "Postfix: " << postfix << " || ";

    // LinkedBinaryTree t = createExpressionTree("a b + b *");
    LinkedBinaryTree t = createExpressionTree(postfix);
    // t.printExpression();
    // cout << endl;
    return t;
}

// Evaluate tree t in the rocket centering task
void evaluate(mt19937& rng, LinkedBinaryTree& t, const int& num_episode, bool animate) {
    rocketCentering env;
    double mean_score = 0.0;
    double mean_steps = 0.0;

    for (int i = 0; i < num_episode; i++) {
        double episode_score = 0.0;
        int episode_steps = 0;
        env.reset(rng);
        while (!env.terminal()) {
            int action = t.evaluateExpression(env.getRocketXPos(), env.getRocketXVel());
            episode_score += env.update(action, animate);
            episode_steps++;
        }
        mean_score += episode_score;
        mean_steps += episode_steps;
    }

    t.setScore(mean_score / num_episode);
    t.setSteps(mean_steps / num_episode);
}

// Crossover function
void LinkedBinaryTree::crossover(LinkedBinaryTree& partner, mt19937& rng) {
    // Determine which subtrees to swap
    Node* this_node = positions()[randInt(rng, 1, size()-1)].v;
    Node* partner_node = partner.positions()[randInt(rng, 1, partner.size()-1)].v;

    // Create a temporary tree to store the this subtree
    LinkedBinaryTree subtree;
    subtree._root = copyPreOrder(this_node);
    Node* this_parent_node = this_node->par;
    deallocTree(this_node);

    // Add the partner subtree to the this tree
    if (this_parent_node->left == nullptr) {
        addLeftChild(Position(this_parent_node), partner_node);
    } else {
        addRightChild(Position(this_parent_node), partner_node);
    }

    // Add the this subtree to the partner tree
    Node* partner_parent_node = partner_node->par;
    deallocTree(partner_node);

    // Add the this subtree to the partner tree
    if (partner_parent_node->left == nullptr) {
        addLeftChild(Position(partner_parent_node), subtree.root());
    } else {
        addRightChild(Position(partner_parent_node), subtree.root());
    }
}

// Deallocate a subtree rooted by node
void LinkedBinaryTree::deallocTree(Node* node) {
    if (!node) return;

    // Disconnect the subtree from the tree
    bool is_root = (node == _root);
    if (is_root) {
        _root = nullptr;
    } else {
        Node* parent = node->par;
        if (parent->left == node) {
            parent->left = nullptr;
        } else {
            parent->right = nullptr;
        }
    }
    
    // Deallocate the subtree
    stack<Node*> nodes;
    nodes.push(node);
    while (!nodes.empty()) {
        Node* current = nodes.top();
        nodes.pop();
        if (current->right) nodes.push(current->right);
        if (current->left) nodes.push(current->left);
        delete current;
    }
}

// Main function
int main() {
    mt19937 rng(42);
    // Experiment parameters
    const int NUM_TREE = 50;
    const int MAX_DEPTH_INITIAL = 1;
    const int MAX_DEPTH = 20;
    const int NUM_EPISODE = 20;
    const int MAX_GENERATIONS = 100;
    const double CROSSOVER_RATE = 0.25;

    // Create an initial "population" of expression trees
    vector<LinkedBinaryTree> trees;
    for (int i = 0; i < NUM_TREE; i++) {
        LinkedBinaryTree t = createRandExpressionTree(MAX_DEPTH_INITIAL, rng);
        trees.push_back(t);
    }

    // Genetic Algorithm loop
    LinkedBinaryTree best_tree;
    cout << "generation,fitness,steps,size,depth" << endl;
    for (int g = 1; g <= MAX_GENERATIONS; g++) {
        // Fitness evaluation
        for (auto& t : trees) {
            if (t.getGeneration() < g - 1) continue;  // skip if not new
            evaluate(rng, t, NUM_EPISODE, false);
        }

        // sort trees using overloaded "<" op (worst->best)
        sort(trees.begin(), trees.end());

        // // sort trees using comparator class (worst->best)
        // sort(trees.begin(), trees.end(), LexLessThan());

        // erase worst 50% of trees (first half of vector)
        trees.erase(trees.begin(), trees.begin() + NUM_TREE / 2);

        // Print stats for best tree
        best_tree = trees[trees.size() - 1];
        cout << g << ",";
        cout << best_tree.getScore() << ",";
        cout << best_tree.getSteps() << ",";
        cout << best_tree.size() << ",";
        cout << best_tree.depth() << endl;

        int i = 1;
        // Selection and mutation
        while (trees.size() < NUM_TREE) {
            // Selected random "parent" tree from survivors
            LinkedBinaryTree parent = trees[randInt(rng, 0, (NUM_TREE / 2) - 1)];
            
            // Create child tree with copy constructor
            LinkedBinaryTree child(parent);
            child.setGeneration(g);
            
            // Mutation
            child.deleteSubtreeMutator(rng); // Delete a randomly selected part of the child's tree
            child.addSubtreeMutator(rng, MAX_DEPTH); // Add a random subtree to the child
            trees.push_back(child);
        }

        // Crossover
        for (int tree_iterator = 0; tree_iterator < NUM_TREE; tree_iterator++) {
            if (randDouble(rng) < CROSSOVER_RATE) {
                LinkedBinaryTree t1 = trees[tree_iterator];
                LinkedBinaryTree t2;
                for (int i = 0; i < NUM_TREE; i++) {
                    t2 = trees[randInt(rng, 0, NUM_TREE - 1)];
                    if (t1.root() != t2.root()) {
                        break;
                    }
                }
                t1.crossover(t2, rng);
            }
        }
    }

    // Evaluate best tree with animation
    const int num_episode = 2;
    evaluate(rng, best_tree, num_episode, true);

    // Print best tree info
    cout << endl << "Best tree:" << endl;
    best_tree.printExpression();
    cout << endl;
    cout << "Generation: " << best_tree.getGeneration() << endl;
    cout << "Size: " << best_tree.size() << endl;
    cout << "Depth: " << best_tree.depth() << endl;
    cout << "Fitness: " << best_tree.getScore() << endl << endl;
}