// C++ program to count occurrences
// of pattern in a text.
#include <iostream>
using namespace std;

void computeLPSArray(string pat, int M, int lps[]);
int KMPSearch(string pat, string txt);
