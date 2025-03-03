#include <iostream>
#include <queue>
#include <vector>

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;

    explicit TreeNode(int x) : val(x), left(nullptr), right(nullptr) {
    }
};

class BinaryTree {
public:
    static TreeNode *createTree(const std::vector<int> &values) {
        if (values.empty())
            return nullptr;

        auto *root = new TreeNode(values[0]);
        std::queue<TreeNode *> q;
        q.push(root);
        int i = 1;
        while (i < values.size()) {
            TreeNode *current = q.front();
            q.pop();

            // assign the left child
            if (values[i] != -1) {
                // assuming -1 mean null
                current->left = new TreeNode(values[i]);
                q.push(current->left);
            }

            i++;

            // assign the right child
            if (i < values.size() && values[i] != -1) {
                current->right = new TreeNode(values[i]);
                q.push(current->right);
            }
            i++;
        }
        return root;
    }

    // function to reverse the binary tree
    static void reverseTree(TreeNode *root) { // NOLINT(*-no-recursion)
        if (root == nullptr)
            return;
        std::swap(root->left, root->right);

        reverseTree(root->left);
        reverseTree(root->right);
    }

    // function to print the tree preorder for verification
    static void preorderTraversal(const TreeNode *root) { // NOLINT(*-no-recursion)
        if (root == nullptr)
            return;
        std::cout << root->val << " ";
        preorderTraversal(root->left);
        preorderTraversal(root->right);
    }
};

int main(int argc, char *argv[]) {
    const std::vector<int> values = {1, 2, 3, 4, 5, 6, 7};

    TreeNode *root = BinaryTree::createTree(values);
    BinaryTree::preorderTraversal(root);

    // reverse the tree
    BinaryTree::reverseTree(root);
    std::cout << std::endl;
    std::cout << "Reversed tree (preorder)" << std::endl;
    BinaryTree::preorderTraversal(root);
    delete root;
    return EXIT_SUCCESS;
}
