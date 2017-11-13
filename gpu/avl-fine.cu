#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct node {
	int data;
	struct node *parent;
	struct node *left;
	struct node *right;
	int height;
	int sema;
} node;

__device__ node* global_root = NULL;

__device__ volatile int MASTER_LOCK = 0;

__device__ int lock(node* n) {
	int status = atomicExch(&n->sema, 1);
	return (!status && !MASTER_LOCK);
}

__device__ void unlock(node* n) {
	atomicExch(&n->sema, 0);
}

__device__ node* new_node(int val, node* parent) {
	node *tmp = (node *) malloc(sizeof(node));
	tmp->data = val;
	tmp->parent = parent;
	tmp->left = tmp->right = NULL;
	tmp->height = 1;
	tmp->sema = 0;
	return tmp;
}

__device__ node* find(node* root, int key) {
	if (root == NULL) return NULL;

	if (root->data == key) return root;
	else if (root->data > key) return find(root->left, key);
	else return find(root->right, key);
}

__device__ int height(node *root)
{
  if (root == NULL)
      return 0;
  return root->height;
}

__device__ int get_balance(node *root)
{
  if (root == NULL)
      return 0;
  return height(root->left) - height(root->right);
}


node* left_rotate(node* root, node* parent)
{
  node* temp1 = root->right;
  node* temp2 = temp1->left;

  temp1->left = root;
  root->parent = temp1;
  root->right = temp2;

  if (temp2)
  	temp2->parent = root;

  root->height = max(height(root->left), height(root->right))+1;
  temp1->height = max(height(temp1->left), height(temp1->right))+1;

  temp1->parent = parent;
  return temp1;
}

node* right_rotate(node* root, node* parent)
{
  node* temp1 = root->left;
  node* temp2 = temp1->right;

  temp1->right = root;
  root->parent = temp1;
  root->left = temp2;

  if(temp2)
  	temp2->parent = root;

  root->height = max(height(root->left), height(root->right))+1;
  temp1->height = max(height(temp1->left), height(temp1->right))+1;

  temp1->parent = parent;
  return temp1;
}


__device__ void rebalance(node* root, int key) {
	root->height = max(height(root->left),height(root->right)) + 1;
	int balance = get_balance(root);

	// Left Left Case
  if (balance > 1 && key < root->left->data) {
		if (root->parent)
			root->parent = right_rotate(root, root->parent);
		else 
			global_root = right_rotate(root, global_root);
	}

	// Right Right Case
  else if (balance < -1 && key > root->right->data) {
  	if (root->parent)
  		root->parent = left_rotate(root, root->parent);
  	else
  		global_root = right_rotate(root, global_root);
  }

	// Left Right Case
  else if (balance > 1 && key > root->left->data) {
  	root->left =  left_rotate(root->left, root);
  	if (root->parent)
  		root->parent = right_rotate(root, root->parent);
  	else 
  		global_root = right_rotate(root, global_root);
  }

	// Right Left Case
  else if (balance < -1 && key < root->right->data)
  {
		root->right = right_rotate(root->right);
		if (root->parent)
			root->right = left_rotate(root, root->parent);
		else
			global_root = right_rotate(root, global_root);
  }
  else {
  	if (root->parent)
  		rebalance(root->parent, key);
  }
  return;
}


__device__ void insert(node* root, int key) {

	if (root == NULL) { 		 				// Empty Tree
		root = new_node(key, NULL); 
		return;
	}
	
	int acquired = lock(root);

	if (acquired) {
		if (key < root->data) {
			if (root->left == NULL) {			// Can be inserted to the immediate left
				root->left = new_node(key, root);
				unlock(root);
				while (!atomicExch(&MASTER_LOCK, 1));
				rebalance(root, key);
				atomicExch(&MASTER_LOCK, 0);
				return;
			} else {											// Release this Node and proceed
				unlock(root);
				insert(root->left, key);
			}
		} else {
			if (root->right == NULL) {		// Can be inserted to the immediate right
				root->right = new_node(key, root);
				unlock(root);
				while (!atomicExch(&MASTER_LOCK, 1));
				rebalance(root, key);
				atomicExch(&MASTER_LOCK, 0);
				return;
			} else {
				unlock(root);								// Release this Node and proceed
				insert(root->right, key);
			}
		}
		

	} else {
		insert(root, key);
	}
}


__device__ void pre_order(node* root)
{
    if(root != NULL)
    {
      printf("%d ", root->data);
      pre_order(root->left);
      pre_order(root->right);
    }
		return;
}

__device__ void in_order(node* root)
{
    if(root != NULL)
    {
      in_order(root->left);
      printf("%d ", root->data);
      in_order(root->right);
    }
		return;
}

/*
__device__ node* min_BST(node* root) {
	node* tmp = root->right;
	if (tmp == NULL) return NULL;

	while(tmp->left != NULL)	tmp = tmp->left;
	return tmp;
}

__device__ void bst_delete(node* root, int key) {
	if (root == NULL) return;
	// printf("del key= %d\n", key);
	//int root_acquired = lock(root);
	int root_acquired = 1;
	if (root_acquired) {
		node* node2delete = find(root, key);

		if (node2delete) {
			//printf("Delete Node %d\n",node2delete->data);
			node* parent = node2delete->parent;
			if (parent) {
				//unlock(root);
				int parent_acquired = lock(parent);
				if (parent_acquired) {
					node* successor = min_BST(node2delete);
					if (successor == NULL) {										// Leaf Node
						if (node2delete->data < parent->data) {
							parent->left = node2delete->left;
						} else {
							parent->right = node2delete->left;
						}
						if(node2delete->left)
							node2delete->left->parent = parent;
						free(node2delete);
					} 
					else if (successor != NULL) {
						node* parent_of_successor = successor->parent;
						node2delete->data = successor->data;
						if (successor->data < parent_of_successor->data) {
							parent_of_successor->left = successor->right;
						} else {
							parent_of_successor->right = successor->right;
						}
						if(successor->right)
							successor->right->parent = parent_of_successor;
						free(successor);
					}
					unlock(parent);
				} else {
					//printf("recall %d\n", key);
					bst_delete(root, key);
				}
			} else {												// ROOT of tree involved!
				// not handled!
			}
		} else {
			//unlock(root);
		}
	} else {
		//printf("recall %d\n", key);
		bst_delete(root, key);
	}
}
*/