#include <stdio.h>
#include <stdlib.h>
#include "timer.h"
#include "../cpu/bst.c"

node* root;

void insert_N(int N) {
	int i;
	for (i=0; i<N; i++)
		root = insert(root, i);
}

void delete_N(int N) {
	int i;
	for (i=0; i<N; i++)
		if (i%4 == 1)
			root = bst_delete(root, i);
}

void find_N(int N) {
	int i;
	for (i=0; i<N; i++)
		find(root, i);
}

int main(void) {	
	CPUTimer time_insert, time_delete, time_find;
	time_insert.Start();
	insert_N(10);
	time_insert.Stop();

	time_delete.Start();
	delete_N(10);
	time_delete.Stop();

	time_find.Start();
	find_N(10);
	time_find.Stop();

	printf("Insert took: %f ms\n", time_insert.Elapsed());
	printf("Delete took: %f ms\n", time_delete.Elapsed());
	printf("Find   took: %f ms\n", time_find.Elapsed());
	
	in_order(root);
	printf("\n");
	return 0;
}
