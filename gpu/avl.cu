#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

typedef struct node {
	int data;
	struct node *parent;
	struct node *left;
	struct node *right;
	int sema;
	int height;
} node;

/*
__device__ int max(int a, int b)
{
    if(a > b)
			return a;
		return b;
}
*/

__device__ int lock(node* n) {
	return !atomicExch(&n->sema, 1);
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
	return tmp;
}

__device__ int height(node *root)
{
    if (root == NULL)
        return 0;
    return root->height;
}

__device__ node* left_rotate(node* root)
{
    node* temp1 = root->right;
    node* temp2 = temp1->left;
 
    temp1->left = root;
    root->right = temp2;
 
    root->height = max(height(root->left), height(root->right))+1;
    temp1->height = max(height(temp1->left), height(temp1->right))+1;
 
    return temp1;
}

__device__ node* right_rotate(node* root)
{
    node* temp1 = root->left;
    node* temp2 = temp1->right;
 
    temp1->right = root;
    root->left = temp2;
 
    root->height = max(height(root->left), height(root->right))+1;
    temp1->height = max(height(temp1->left), height(temp1->right))+1;
 
    return temp1;
}

__device__ int get_balance(node *root)
{
    if (root == NULL)
        return 0;
    return height(root->left) - height(root->right);
}

__device__ int MASTER_LOCK = 0;

__device__ void rebalance(node* p, int key) {
	bool flag = true;
	if (p->parent) {
		while (atomicExch(&(p->parent->sema), 1) && flag) {
			// acquired 
			p->height = max(height(p->left), height(p->right)) + 1;
			int balance = get_balance(p);

			if (balance > 1 && key < p->left->data) {
				if (p->data < p->parent->data) {
					p->parent->left = right_rotate(p);
				} else {
					p->parent->right = right_rotate(p);
				}
			}

			// Right Right Case
		  if (balance < -1 && key > p->right->data) {
		  	if (p->data < p->parent->data) {
		  		p->parent->left = left_rotate(p);
		  	} else {
		  		p->parent->right = left_rotate(p);
		  	}
		  }

			// Left Right Case
		  if (balance > 1 && key > p->left->data)
		  {
		  	p->left =  left_rotate(p->left);

		  	if (p->data < p->parent->data) {
					p->parent->left = right_rotate(p);
				} else {
					p->parent->right = right_rotate(p);
				}
		  }

			// Right Left Case
		  if (balance < -1 && key < p->right->data)
		  {
				p->right = right_rotate(p->right);
				
				if (p->data < p->parent->data) {
		  		p->parent->left = left_rotate(p);
		  	} else {
		  		p->parent->right = left_rotate(p);
		  	}
		  }


			atomicExch(&(p->parent->sema), 0);
			flag = false;
			rebalance(p->parent, key);
		}
	} else {																						// ROOT balance
		while(atomicExch(&MASTER_LOCK, 1) && flag) {

			p->height = max(height(p->left), height(p->right)) + 1;
			int balance = get_balance(p);

			if (balance > 1 && key < p->left->data) {
				if (p->data < p->parent->data) {
					p->parent->left = right_rotate(p);
				} else {
					p->parent->right = right_rotate(p);
				}
			}

			// Right Right Case
		  if (balance < -1 && key > p->right->data) {
		  	if (p->data < p->parent->data) {
		  		p->parent->left = left_rotate(p);
		  	} else {
		  		p->parent->right = left_rotate(p);
		  	}
		  }

			// Left Right Case
		  if (balance > 1 && key > p->left->data)
		  {
		  	p->left =  left_rotate(p->left);

		  	if (p->data < p->parent->data) {
					p->parent->left = right_rotate(p);
				} else {
					p->parent->right = right_rotate(p);
				}
		  }

			// Right Left Case
		  if (balance < -1 && key < p->right->data)
		  {
				p->right = right_rotate(p->right);
				
				if (p->data < p->parent->data) {
		  		p->parent->left = left_rotate(p);
		  	} else {
		  		p->parent->right = left_rotate(p);
		  	}
		  }

			atomicExch(&MASTER_LOCK, 0);
			flag = false;
		}
	}
}

__device__ void coarse_rebalance(node* p, int key) {
	if (p->parent) {
		p->height = max(height(p->left), height(p->right)) + 1;
		int balance = get_balance(p);

		if (balance > 1 && key < p->left->data) {
			if (p->data < p->parent->data) {
				p->parent->left = right_rotate(p);
			} else {
				p->parent->right = right_rotate(p);
			}
		}

		// Right Right Case
	  if (balance < -1 && key > p->right->data) {
	  	if (p->data < p->parent->data) {
	  		p->parent->left = left_rotate(p);
	  	} else {
	  		p->parent->right = left_rotate(p);
	  	}
	  }

		// Left Right Case
	  if (balance > 1 && key > p->left->data)
	  {
	  	p->left =  left_rotate(p->left);

	  	if (p->data < p->parent->data) {
				p->parent->left = right_rotate(p);
			} else {
				p->parent->right = right_rotate(p);
			}
	  }

		// Right Left Case
	  if (balance < -1 && key < p->right->data)
	  {
			p->right = right_rotate(p->right);
			
			if (p->data < p->parent->data) {
	  		p->parent->left = left_rotate(p);
	  	} else {
	  		p->parent->right = left_rotate(p);
	  	}
	  }

		coarse_rebalance(p->parent, key);
	}
}


__device__ void coarse_insert(node* root, int key) {

	node* curr = root;
	node* parent = NULL;
	
	if (root == NULL) {
		root = new_node(key, parent);
		return;
	}

	bool flag = true;
	while (flag) {
		if (!atomicExch(&MASTER_LOCK, 1)) {
			while (curr != NULL) {	
				parent = curr;
				if (key < curr->data)
					curr = curr->left;
				else
					curr = curr->right;	
				if (curr == NULL) {
					if (key < parent->data) {
						parent->left = new_node(key, parent);
						coarse_insert(parent->left, key);
					} else {
						parent->right = new_node(key, parent);
						coarse_insert(parent->right, key);
					}			
				} else {
					if (parent)
						atomicExch(&(parent->sema), 0);
				}
			}
			flag = false;
			atomicExch(&MASTER_LOCK, 0);
		}
	}	
}

__device__ void coarse_delete(node* root, int key) {
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
				return;
			} else {											// Release this Node and proceed
				unlock(root);
				insert(root->left, key);
			}
		} else {
			if (root->right == NULL) {		// Can be inserted to the immediate right
				root->right = new_node(key, root);
				unlock(root);
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

__device__ node* find(node* root, int key) {
	if (root == NULL) return NULL;

	if (root->data == key) return root;
	else if (root->data > key) return find(root->left, key);
	else return find(root->right, key);
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

// __device__ node* min_BST(node* Node) {
// 	if (Node == NULL) return NULL;
// 	node* tmp = Node;
// 	while(tmp->left != NULL)	tmp = tmp->left;
// 	return tmp;
// }

// __device__ node* delete(node* root, int key) {
// 	if (root == NULL) return NULL;

// 	if (key < root->data) 
// 		root->left  = delete(root->left, key);
// 	else if (key > root->data) 
// 		root->right = delete(root->right, key); 
// 	else {
// 		if (root->left == NULL) {
// 			node* tmp = root->right;
// 			free(root);
// 			return tmp;
// 		} 
// 		else if (root->right == NULL) {
// 			node* tmp = root->left;
// 			free(root);
// 			return tmp;
// 		}
// 		// successor
// 		node *tmp = min_BST(root->right);
// 		root->data = tmp->data;
// 		root->right = delete(root->right, tmp->data);
// 	}

// 	if (root == NULL)
//       return root;

// 	root->height = max(height(root->left),height(root->right)) + 1;

// 	int balance = get_balance(root);

// 	// Left Left Case
//   if (balance > 1 && get_balance(root->left) >= 0)
//   	return right_rotate(root);
 
//   // Left Right Case
// 	if (balance > 1 && get_balance(root->left) < 0)
//   {
//   	root->left =  left_rotate(root->left);
//   	return right_rotate(root);
//   }
 
//   // Right Right Case
//   if (balance < -1 && get_balance(root->right) <= 0)
//   	return left_rotate(root);
 
//   // Right Left Case
//   if (balance < -1 && get_balance(root->right) > 0)
//   {
//   	root->right = right_rotate(root->right);
//   	return left_rotate(root);
//   }
 
// 	return root;
// }
