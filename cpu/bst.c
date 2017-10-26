/* @auth : Arjun Krishna
 * @desc : Non-concurrent implementation of BST
 					 [Runs on CPU]
 */

#include <stdio.h>
#include <stdlib.h>


typedef struct node {
	int data;
	struct node *left;
	struct node *right;
} node;


node* new_node(int val) {
	node *tmp = (node *) malloc(sizeof(node));
	tmp->data = val;
	tmp->left = tmp->right = NULL;
	return tmp;
}

node* insert(node* root, int key) {
	if (root == NULL) return new_node(key);

	if (key < root->data) 
		root->left  = insert(root->left, key);
	else 
		root->right = insert(root->right, key);

	return root;
}

node* min_BST(node* root) {
	if (root == NULL) return NULL;
	node* tmp = root;
	while(tmp->left != NULL)	tmp = tmp->left;
	return tmp;
}

node* delete(node* root, int key) {
	if (root == NULL) return NULL;

	if (key < root->data) 
		root->left  = delete(root->left, key);
	else if (key > root->data) 
		root->right = delete(root->right, key); 
	else {
		if (root->left == NULL) {
			node* tmp = root->right;
			free(root);
			return tmp;
		} 
		else if (root->right == NULL) {
			node* tmp = root->left;
			free(root);
			return tmp;
		}
		// successor
		node* tmp = min_BST(root->right);
		root->data = tmp->data;
		root->right = delete(root->right, tmp->data);
	}
	return root;
}

int main(void) {
	return 0;
}