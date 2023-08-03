#include <cryptominisat5/cryptominisat.h>
#include <assert.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string.h>
#include <vector>

#define FILENAME "/home/ange/WORKSPACE/AI/ml-in-action/constraint-learning/data/nsl-kdd.cnf"
#define FILENAME2 "/home/ange/WORKSPACE/AI/ml-in-action/constraint-learning/test/3-replicate.cnf"

using namespace CMSat;
using namespace std;

vector<int> parse_cast_and_convert_to_vect(string const &str)
{
  // Create a vector to store integers
  vector<int> vect;

  // Iterate over the string and parse each integer
  for (int i = 0; i < str.length(); i++)
    // Check if the current character is a digit or the negative signe (-)
    if (isdigit(str[i]) || str[i] == '-')
    {
      // Create a string to store the current integer
      string num = "";

      // Iterate over the next few characters and append them to the string
      for (int j = i; j < str.length() && (isdigit(str[j]) || str[j] == '-'); j++)
      {
        num += str[j];
        i = j;
      }

      // Convert the string to an integer
      stringstream ss;
      int value = 0;
      ss << num;
      ss >> value;

      // exclude ends zeros
      if (value == 0)
        continue;

      // store it in the vector
      vect.push_back(value);
    }

  /** Print the vector */
  // for (int i = 0; i < vect.size(); i++)
  //   cout << vect[i] << " ";
  // cout << endl;

  return vect;
}

vector<vector<int>> readfile(string const &filename)
{
  // Create a 2D vector of shape (line, columns) = (20, 2318)
  // vector<vector<int>> data(0, vector<int>(VARIABLES));
  vector<vector<int>> data;

  // Open the file
  fstream file(filename);

  // Iterate from the 2nd line of the file to the end
  if (file)
  {
    string line = "";

    /** Get shape at the 1st line of the file */
    getline(file, line);
    vector<int> shape = parse_cast_and_convert_to_vect(line);
    int variables = shape[0],
        clauses = shape[1];
    cout << clauses << " * " << variables << endl;

    // 2nd line of the file
    while (getline(file, line))
      // Store the line as a row of the 2D vector
      data.push_back(parse_cast_and_convert_to_vect(line));
  }
  else
    cout << "ERROR." << endl;

  // Close the file
  file.close();

  /** show size of each row*/
  // cout << data.size() << endl;
  // for (int i = 0; i < data.size(); i++)
  //   cout << data[i].size() << " | ";
  // cout << endl;

  return data;
}

vector<int> get_shape(vector<vector<int>> const &mat)
{
  int rows = mat.size(), columns = 0;

  // Maximum
  for (int i = 0; i < mat.size(); i++)
    columns = columns < mat[i].size() ? mat[i].size() : columns;

  return vector<int>{rows, columns};
}

void print_satisfiable_sample(SATSolver const &solver)
{
  cout << "Solution is: \t" << solver.get_model()[0];
  for (int i = 1; i < solver.nVars(); i++)
    cout << ", " << solver.get_model()[i];
  cout << endl;
}

void satTest()
{
  SATSolver solver;
  vector<Lit> clause;

  // Let's use 4 threads
  solver.set_num_threads(4);

  // We need 3 variables. They will be: 0,1,2
  // Variable numbers are always trivially increasing
  solver.new_vars(3);
  lbool ret;

  /*******************************************************************************/
  vector<vector<int>> cnf_arr{
      {1},         // "1 0"
      {-2},        // "-2 0"
      {-1, 2, 3}}; // "-1 2 3 0"

  for (int i = 0; i < cnf_arr.size(); i++)
  {
    clause.clear();
    for (int j = 0; j < cnf_arr[i].size(); j++)
    {
      // clause.push_back(Lit(0, false));
      clause.push_back(Lit(abs(cnf_arr[i][j]) - 1, cnf_arr[i][j] < 0));
    }
    solver.add_clause(clause);
  }

  /** ****************************************************************************
   *                                 SATISFIABLE                                 *
  /***************************************************************************** */
  {
    // ret = solver.solve();
    // assert(ret == l_True);
    // cout << "Solution is: "
    //      << solver.get_model()[0]
    //      << ", " << solver.get_model()[1]
    //      << ", " << solver.get_model()[2]
    //      << endl;

    /******************************************************************************/
    int a = 0;
    while (true)
    {
      lbool ret = solver.solve();
      if (ret != l_True)
      {
        assert(ret == l_False);
        // All solutions found.
        exit(0);
      }

      // Use solution here. print it, for example.
      assert(ret == l_True);
      cout << "Solution is: "
           << solver.get_model()[0]
           << ", " << solver.get_model()[1]
           << ", " << solver.get_model()[2]
           << endl;

      cout << "solver.nVars() |----->  " << solver.nVars() << endl;
      cout << "a |----->  " << ++a << endl;
      // Banning found solution
      vector<Lit> ban_solution;
      for (uint32_t var = 0; var < solver.nVars(); var++)
      {
        if (solver.get_model()[var] != l_Undef)
        {
          ban_solution.push_back(
              Lit(var, (solver.get_model()[var] == l_True) ? true : false));
        }
      }
      solver.add_clause(ban_solution);
    }
    /******************************************************************************/
  }

  /** ****************************************************************************
   *                                 ASSUMPTION                                  *
  /***************************************************************************** */
  {
    // --> assumes 3 = FALSE, no solutions left
    vector<Lit> assumptions;
    assumptions.push_back(Lit(2, true));
    ret = solver.solve(&assumptions, true);
    assert(ret == l_False); // UNSATISFIABLE

    // --> without assumptions we still have a solution
    ret = solver.solve();
    assert(ret == l_True);
  }

  /** ****************************************************************************
   *                                UNSATISFIABLE                                *
  /***************************************************************************** */
  {
    // add "-3 0"
    // No solutions left, UNSATISFIABLE returned
    clause.clear();
    clause.push_back(Lit(2, true));
    solver.add_clause(clause);
    ret = solver.solve();
    assert(ret == l_False);
  }
}

void sat()
{
  SATSolver solver;
  vector<Lit> clause;
  lbool ret;
  int count = 0;
  vector<vector<int>> cnf_arr = readfile(FILENAME);

  // Shape ++ Variables size definition
  vector<int> shape = get_shape(cnf_arr);
  int threads = shape[0],
      variables = shape[1];
  cout << threads << " - " << variables << endl;
  solver.set_num_threads(threads);
  solver.new_vars(variables);
  return;

  /*******************************************************************************/
  for (int i = 0; i < threads; i++)
  {
    clause.clear();

    for (int j = 0; j < cnf_arr[i].size(); j++)
      clause.push_back(Lit(abs(cnf_arr[i][j]) - 1, cnf_arr[i][j] < 0));

    solver.add_clause(clause);
  }

  /*******************************************************************************/
  /** show only one satisfiable sample*/
  // assert(ret == l_True);
  // ret = solver.solve();
  // print_satisfiable_sample(solver);
  // return;

  while (true)
  {
    ret = solver.solve();

    if (ret != l_True)
    {
      // All solutions found.
      assert(ret == l_False);
      cout << "No solution" << endl;
      exit(0);
    }

    /** ELSE */
    assert(ret == l_True);
    count++;

    /** show satisfiable sample*/
    // print_satisfiable_sample(solver);
    cout << "# sample |----->  " << count << endl;

    /** Banning found solution */
    vector<Lit> ban_solution;

    for (uint32_t var = 0; var < solver.nVars(); var++)
      if (solver.get_model()[var] != l_Undef)
        ban_solution.push_back(Lit(var, (solver.get_model()[var] == l_True) ? true : false));

    solver.add_clause(ban_solution);
  }
  /*******************************************************************************/
}

int main()
{
  sat();
  // satTest();

  return 0;
}
