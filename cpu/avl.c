#include <stdio.h>
#include <stdlib.h>


typedef struct node {
	int data;
	struct node *left;
	struct node *right;
	int height;
} node;

int max(int a, int b)
{
    if(a > b)
			return a;
		return b;
}

node* new_node(int val) {
	node *tmp = (node *) malloc(sizeof(node));
	tmp->data = val;
	tmp->left = tmp->right = NULL;
	tmp->height = 1;
	return tmp;
}

int height(node *Node)
{
    if (Node == NULL)
        return 0;
    return Node->height;
}

node* leftRotate(node* Node)
{
    node* temp1 = Node->right;
    node* temp2 = temp1->left;
 
    temp1->left = Node;
    Node->right = temp2;
 
    Node->height = max(height(Node->left), height(Node->right))+1;
    temp1->height = max(height(temp1->left), height(temp1->right))+1;
 
    return temp1;
}

node* rightRotate(node* Node)
{
    node* temp1 = Node->left;
    node* temp2 = temp1->right;
 
    temp1->right = Node;
    Node->left = temp2;
 
    Node->height = max(height(Node->left), height(Node->right))+1;
    temp1->height = max(height(temp1->left), height(temp1->right))+1;
 
    return temp1;
}

int getBalance(node *Node)
{
    if (Node == NULL)
        return 0;
    return height(Node->left) - height(Node->right);
}

node* insert(node* Node, int key) {

	//Normal BST key insertion
	if (Node == NULL) return new_node(key);

	if (key < Node->data) 
		Node->left  = insert(Node->left, key);
	else 
		Node->right = insert(Node->right, key);

	Node->height = max(height(Node->left),height(Node->right)) + 1;
	int balance = getBalance(Node);

	// Left Left Case
  if (balance > 1 && key < Node->left->data)
		return rightRotate(Node);

	// Right Right Case
  if (balance < -1 && key > Node->right->data)
  	return leftRotate(Node);

	// Left Right Case
  if (balance > 1 && key > Node->left->data)
  {
  	Node->left =  leftRotate(Node->left);
  	return rightRotate(Node);
  }

	// Right Left Case
  if (balance < -1 && key < Node->right->data)
  {
		Node->right = rightRotate(Node->right);
		return leftRotate(Node);
  }

	//If Node is balanced
	return Node;
}

void preOrder(node* Node)
{
    if(Node != NULL)
    {
        printf("%d ", Node->data);
        preOrder(Node->left);
        preOrder(Node->right);
    }
		return;
}

node* min_BST(node* Node) {
	if (Node == NULL) return NULL;
	node* tmp = Node;
	while(tmp->left != NULL)	tmp = tmp->left;
	return tmp;
}

node* delete(node* Node, int key) {
	if (Node == NULL) return NULL;

	if (key < Node->data) 
		Node->left  = delete(Node->left, key);
	else if (key > Node->data) 
		Node->right = delete(Node->right, key); 
	else {
		if (Node->left == NULL) {
			node* tmp = Node->right;
			free(Node);
			return tmp;
		} 
		else if (Node->right == NULL) {
			node* tmp = Node->left;
			free(Node);
			return tmp;
		}
		// successor
		node *tmp = min_BST(Node->right);
		Node->data = tmp->data;
		Node->right = delete(Node->right, tmp->data);
	}

	if (Node == NULL)
      return Node;

	Node->height = max(height(Node->left),height(Node->right)) + 1;

	int balance = getBalance(Node);

	// Left Left Case
  if (balance > 1 && getBalance(Node->left) >= 0)
  	return rightRotate(Node);
 
  // Left Right Case
	if (balance > 1 && getBalance(Node->left) < 0)
  {
  	Node->left =  leftRotate(Node->left);
  	return rightRotate(Node);
  }
 
  // Right Right Case
  if (balance < -1 && getBalance(Node->right) <= 0)
  	return leftRotate(Node);
 
  // Right Left Case
  if (balance < -1 && getBalance(Node->right) > 0)
  {
  	Node->right = rightRotate(Node->right);
  	return leftRotate(Node);
  }
 
	return Node;
}


int main()
{
  return 0;
}
