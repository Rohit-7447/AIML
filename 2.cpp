// Movie Recommendation System using QuickSort
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
using namespace std;

class Movie {
public:
    string title;
    double imdbRating;
    int releaseYear;
    int watchCount;

    Movie(string t, double r, int y, int w)
        : title(t), imdbRating(r), releaseYear(y), watchCount(w) {}
};

// Partition function for QuickSort
int partition ( vector<Movie> & movies,int low,int high,const string &sortBy){
    Movie pivot =movies[low];
    int i=low+1;
    int j=high;
    while (i<=j){
        if (sortBy =="rating"){
            while (i<=high && movies[i].imdbRating<=pivot.imdbRating) i++;
            while (j>=low && movies[j].imdbRating>pivot.imdbRating) j--;
        }
        else if (sortBy == "year") {
            while (i <= high && movies[i].releaseYear <= pivot.releaseYear) i++;
            while (j >= low && movies[j].releaseYear > pivot.releaseYear) j--;
        } else if (sortBy == "watch") {
            while (i <= high && movies[i].watchCount <= pivot.watchCount) i++;
            while (j >= low && movies[j].watchCount > pivot.watchCount) j--;
        }
        if (i< j){
            swap(movies[i],movies[j]);
        }

    }
    swap(movies[low],movies[j]);
    return j;

}

// QuickSort function
void quickSort(vector<Movie>& movies, int low, int high, const string& sortBy) {
    if (low < high) {
        int pivotIndex = partition(movies, low, high, sortBy);
        quickSort(movies, low, pivotIndex - 1, sortBy);
        quickSort(movies, pivotIndex + 1, high, sortBy);
    }
}

// Sample movie data
vector<Movie> sampleMovies() {
    return {
        {"Avengers", 8.5, 2012, 9500},
        {"Inception", 8.8, 2010, 8700},
        {"Titanic", 7.8, 1997, 12000},
        {"Matrix", 8.7, 1999, 9800},
        {"Interstellar", 8.6, 2014, 7300},
        {"Joker", 8.4, 2019, 8200},
        {"Avatar", 7.9, 2009, 15000},
        {"Gladiator", 8.5, 2000, 6600},
        {"Up", 8.2, 2009, 5400},
        {"Coco", 8.4, 2017, 4700}
    };
}

int main() {
    vector<Movie> movies = sampleMovies();
    string sortBy;

    cout << "Sort movies by (rating/year/watch): ";
    cin >> sortBy;

    transform(sortBy.begin(), sortBy.end(), sortBy.begin(), ::tolower);
    if (sortBy != "rating" && sortBy != "year" && sortBy != "watch") {
        cout << "Invalid sort option!" << endl;
        return 0;
    }

    auto start = chrono::high_resolution_clock::now();
    quickSort(movies, 0, movies.size() - 1, sortBy);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> timeTaken = end - start;

    cout << "\nMovies sorted by " << sortBy << ":\n";
    printf("%-15s %-8s %-6s %-6s\n", "Title", "Rating", "Year", "Watch");
    cout << "-------------------------------------------\n";
    for (const auto& m : movies) {
        printf("%-15s %-8.1f %-6d %-6d\n", m.title.c_str(), m.imdbRating, m.releaseYear, m.watchCount);
    }
    printf("\nTime taken by QuickSort: %.3f ms\n", timeTaken.count());
    return 0;
}



// //Method : Taking input as csv from file==============
// #include <iostream>
// #include <vector>
// #include <string>
// #include <fstream>
// #include <sstream>
// #include <chrono>
// #include <algorithm>
// using namespace std;

// class Movie {
// public:
//     string title;
//     double imdbRating;
//     int releaseYear;
//     int watchCount;

//     Movie(string t, double r, int y, int w)
//         : title(t), imdbRating(r), releaseYear(y), watchCount(w) {}
// };

// // Partition function for QuickSort
// int partition(vector<Movie>& movies, int low, int high, const string& sortBy) {
//     Movie pivot = movies[low];
//     int i = low + 1;
//     int j = high;
//     while (i <= j) {
//         if (sortBy == "rating") {
//             while (i <= high && movies[i].imdbRating <= pivot.imdbRating) i++;
//             while (j >= low && movies[j].imdbRating > pivot.imdbRating) j--;
//         } else if (sortBy == "year") {
//             while (i <= high && movies[i].releaseYear <= pivot.releaseYear) i++;
//             while (j >= low && movies[j].releaseYear > pivot.releaseYear) j--;
//         } else if (sortBy == "watch") {
//             while (i <= high && movies[i].watchCount <= pivot.watchCount) i++;
//             while (j >= low && movies[j].watchCount > pivot.watchCount) j--;
//         }
//         if (i < j) {
//             swap(movies[i], movies[j]);
//         }
//     }
//     swap(movies[low], movies[j]);
//     return j;
// }

// // QuickSort function
// void quickSort(vector<Movie>& movies, int low, int high, const string& sortBy) {
//     if (low < high) {
//         int pivotIndex = partition(movies, low, high, sortBy);
//         quickSort(movies, low, pivotIndex - 1, sortBy);
//         quickSort(movies, pivotIndex + 1, high, sortBy);
//     }
// }

// // Function to load movies from CSV
// vector<Movie> loadMoviesFromCSV(const string& filename) {
//     vector<Movie> movies;
//     ifstream file(filename);
//     if (!file.is_open()) {
//         cout << "Error: Could not open file " << filename << endl;
//         return movies;
//     }

//     string line;
//     getline(file, line); // skip header

//     while (getline(file, line)) {
//         stringstream ss(line);
//         string title, ratingStr, yearStr, watchStr;

//         getline(ss, title, ',');
//         getline(ss, ratingStr, ',');
//         getline(ss, yearStr, ',');
//         getline(ss, watchStr, ',');

//         double rating = stod(ratingStr);
//         int year = stoi(yearStr);
//         int watchCount = stoi(watchStr);

//         movies.push_back(Movie(title, rating, year, watchCount));
//     }

//     file.close();
//     return movies;
// }

// int main() {
//     string filename = "movies.csv";
//     vector<Movie> movies = loadMoviesFromCSV(filename);

//     if (movies.empty()) {
//         cout << "No movies loaded. Exiting.\n";
//         return 0;
//     }

//     string sortBy;
//     cout << "Sort movies by (rating/year/watch): ";
//     cin >> sortBy;

//     transform(sortBy.begin(), sortBy.end(), sortBy.begin(), ::tolower);
//     if (sortBy != "rating" && sortBy != "year" && sortBy != "watch") {
//         cout << "Invalid sort option!" << endl;
//         return 0;
//     }

//     auto start = chrono::high_resolution_clock::now();
//     quickSort(movies, 0, movies.size() - 1, sortBy);
//     auto end = chrono::high_resolution_clock::now();
//     chrono::duration<double, milli> timeTaken = end - start;

//     cout << "\nMovies sorted by " << sortBy << ":\n";
//     printf("%-15s %-8s %-6s %-6s\n", "Title", "Rating", "Year", "Watch");
//     cout << "-------------------------------------------\n";
//     for (int i = 0; i < movies.size(); i++) {
//         printf("%-15s %-8.1f %-6d %-6d\n",
//                movies[i].title.c_str(),
//                movies[i].imdbRating,
//                movies[i].releaseYear,
//                movies[i].watchCount);
//     }
//     printf("\nTime taken by QuickSort: %.3f ms\n", timeTaken.count());
//     return 0;
// }

//Method 2 Pratik
// #include <bits/stdc++.h>
// #include <iomanip>
// using namespace std;
// using namespace std::chrono;


// // --------------------- MOVIE STRUCTURE ---------------------
// struct Movie {
//     string title;
//     float rating;
//     int release_year;
//     int popularity;


//     void display() const {
//         cout << title
//              << " | Rating: " << rating
//              << " | Year: " << release_year
//              << " | Popularity: " << popularity << endl;
//     }
// };


// // --------------------- QUICKSORT IMPLEMENTATION ---------------------
// void quickSort(vector<Movie> &movies, int low, int high, bool (*compare)(const Movie &, const Movie &)) {
//     if (low < high) {
//         int pivotIndex = low;
//         Movie pivot = movies[high];
//         for (int i = low; i < high; i++) {
//             if (compare(movies[i], pivot)) {
//                 swap(movies[i], movies[pivotIndex]);
//                 pivotIndex++;
//             }
//         }
//         swap(movies[pivotIndex], movies[high]);


//         quickSort(movies, low, pivotIndex - 1, compare);
//         quickSort(movies, pivotIndex + 1, high, compare);
//     }
// }


// // --------------------- COMPARATOR FUNCTIONS ---------------------
// bool compareByRating(const Movie &a, const Movie &b) {
//     return a.rating < b.rating;
// }


// bool compareByYear(const Movie &a, const Movie &b) {
//     return a.release_year < b.release_year;
// }


// bool compareByPopularity(const Movie &a, const Movie &b) {
//     return a.popularity < b.popularity;
// }


// // --------------------- RANDOM MOVIE GENERATOR ---------------------
// vector<Movie> generateMovies(int n) {
//     vector<Movie> movies;
//     movies.reserve(n);
//     for (int i = 0; i < n; i++) {
//         Movie m;
//         m.title = "Movie " + to_string(i + 1);
//         m.rating = static_cast<float>(rand() % 90 + 10) / 10.0f;   // Ratings between 1.0 and 10.0
//         m.release_year = rand() % 45 + 1980;                       // 1980 to 2024
//         m.popularity = rand() % 1000000 + 1000;                    // 1,000 to 1,000,000
//         movies.push_back(m);
//     }
//     return movies;
// }


// // --------------------- MAIN FUNCTION ---------------------
// int main() {
//     srand(time(0));


//     int num_movies = 1000000; // Use 1,000,000 for stress testing
//     cout << "Generating " << num_movies << " random movies..." << endl;


//     vector<Movie> movies = generateMovies(num_movies);


//     cout << "Sort movies by (rating/year/popularity): ";
//     string sort_by;
//     cin >> sort_by;


//     bool (*compare)(const Movie &, const Movie &);
//     if (sort_by == "rating")
//         compare = compareByRating;
//     else if (sort_by == "year")
//         compare = compareByYear;
//     else if (sort_by == "popularity")
//         compare = compareByPopularity;
//     else {
//         cout << "Invalid choice. Defaulting to rating." << endl;
//         compare = compareByRating;
//     }


//     cout << "Sorting movies using QuickSort..." << endl;
//     auto start = high_resolution_clock::now();


//     quickSort(movies, 0, movies.size() - 1, compare);


//     auto end = high_resolution_clock::now();
//     duration<double> diff = end - start;


//     cout << "Sorting completed in " << fixed << setprecision(3)
//          << diff.count() << " seconds." << endl;


//     // Display top 10 movies in descending order (best-ranked first)
//     cout << "\nTop 10 Movies by " << sort_by << ":" << endl;
//     for (int i = num_movies - 1; i >= num_movies - 10; --i)
//         movies[i].display();


//     return 0;
// }