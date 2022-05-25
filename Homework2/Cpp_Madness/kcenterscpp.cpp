#include <iostream>
#include <fstream>
#include <unistd.h>
#include <vector>
#include <immintrin.h>

using namespace std;

/*
vector<float[]> SeqWeightedOutliers(float P[][5], int W[], int k, int z, float alpha)
{
    

    /*
    int attempt = 0;

    // generate matrix with all distances squared
    float all_dist_squared[n][n];
    // supposing dims = 2
    for (int i = 0; i < n; i++)
    {
        float pt_arr[8];
        for (int ind = 0; ind < 8; ind++)
            pt_arr[ind] = (*P)[i][ind%dims];
        
        
        __m256 point = _mm256_load_ps(&pt_arr[0]);  // array containing repetition of current point
        for (int j = 0; j < n / (8 / dims); j++)
        {
            __m256 pts = _mm256_load_ps();
        }
    }*/
    /*if ((dims <= 8) && (8 % dims == 0))
    {
        for (int i = 0; i < n; i++)
        {
            __m256 point = 
            for (int j = 0; j < n / (8 / dims); j++)
            {
                
            }
        }
    }*/
    /*
    
    
    vector<float[]> temp;
    return temp;
    
    

}*/

int main(int argc, char *argv[])
{
    if (argc != 4)
        throw invalid_argument("usage: kcenterscpp <DATASET_PATH> <MAX_N°_OF_CENTERS> <N°_OF_OULIERS");
    
    /*// check if file exists
    if (access(argv[1], F_OK) == -1)
    {
        cout << "file \"" << argv[1] << "\" cannot be found at the specified location" << endl;
        exit(1);
    }*/
    
    // check if second and third arguments are numbers
    float k = 0., z = 0.;
    if (isdigit(*argv[2]))
        k = stof(argv[2]);
    if (isdigit(*argv[3]))
        z = stof(argv[3]);

    
    FILE* f = fopen64(argv[1], "r");
    // does file exists?
    if (f == NULL)
    {
        cout << "file \"" << argv[1] << "\" cannot be found at the specified location" << endl;
        exit(EXIT_FAILURE);
    }
    // load dataset in 2D array
    int n = 0, dims = 0;

    char* line = NULL;
    size_t len = 0;
    if ((getline(&line, &len, f)) != -1)
    {
        for (int i = 0; i < sizeof(line)/sizeof(line[0]); i++)
            if (line[i] == ',')
                dims++;
    }
    n++;
    
    while ((getline(&line, &len, f)) != -1)
        n++;
    

    
    fclose(f);    
}