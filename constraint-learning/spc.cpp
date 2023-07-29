#include <cryptominisat5/cryptominisat.h>
#include <assert.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string.h>
#include <vector>
using namespace CMSat;
using namespace std;

vector<int> parse_cast_and_convert_to_vect(string str)
{
  // Create a vector to store the integers
  vector<int> vect;

  // Iterate over the string and parse each integer
  for (int i = 0; i < str.length(); i++)
  {
    // Check if the current character is a digit
    if (isdigit(str[i]) || str[i] == '-')
    // if (isdigit(str[i]))
    {
      // Create a string to store the current integer
      string num = "";

      // Iterate over the next few characters and append them to the string
      for (int j = i; j < str.length() && (isdigit(str[j]) || str[j] == '-'); j++)
      {
        num += str[j];
        i = j;
      }

      // Convert the string to an integer and store it in the vector
      // int value = stoi(num);
      stringstream ss;
      int value = 0;
      ss << num;
      ss >> value;
      vect.push_back(value);
    }
  }

  // Print the vector
  // for (int i = 0; i < vect.size(); i++)
  // {
  //   cout << vect[i] << " ";
  // }

  return vect;
}

vector<vector<int>> readfile(string filename)
{
  // Create a 2D vector of shape (line, columns) = (20, 2318)
  vector<vector<int>> data(0, vector<int>(2318));

  // Open the file
  fstream file(filename);

  // Iterate from the 2nd line of the file to the end
  if (file)
  {
    string line;
    int i = 0;
    getline(file, line);

    // 2nd line of the file
    while (getline(file, line))
    {
      // if (i > 0)
      //   break;

      // Store the line as a row of the 2D vector
      data.push_back(parse_cast_and_convert_to_vect(line));

      i++;
    }
  }
  else
  {
    cout << "ERROR." << endl;
  }

  // Close the file
  file.close();

  cout << data.size() << endl;
  for (int i = 0; i < data.size(); i++)
    cout << data[i].size() << " | ";

  return data;
}

void sat0()
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
    ret = solver.solve();
    assert(ret == l_True);
    cout << "Solution is: "
         << solver.get_model()[0]
         << ", " << solver.get_model()[1]
         << ", " << solver.get_model()[2]
         << endl;
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
  string filename = "/home/ange/WORKSPACE/AI/ml-in-action/constraint-learning/data/nsl-kdd.cnf";
  vector<vector<int>> cnf_arr = readfile(filename);

  // Shape
  int threads = 20, variables = 2318;
  solver.set_num_threads(threads);
  solver.new_vars(variables);

  /*******************************************************************************/
  for (int i = 0; i < cnf_arr.size(); i++)
  {
    clause.clear();
    for (int j = 0; j < cnf_arr[i].size(); j++)
    {
      if (cnf_arr[i][j] == 0) /** TODO: fix zeros */
        break;
      clause.push_back(Lit(abs(cnf_arr[i][j]) - 1, cnf_arr[i][j] < 0));
    }
    solver.add_clause(clause);
  }

  /*******************************************************************************/
  {
    ret = solver.solve();
    assert(ret == l_True);
    cout << endl
         << "Solution is: "
         << endl;

    cout << solver.get_model()[0];
    for (int i = 1; i < variables; i++)
    {
      cout << ", " << solver.get_model()[i];
    }
    cout << endl;
  }
  /*******************************************************************************/
}

int main()
{
  // parse_cast_and_convert_to_vect("-1 2 3 4 5 6 7 -8 -9 -10 -11 -12 -13 -14 -15 -16 -17 -18 -19 -20 -21 -22 -23 -24 -25 -26 -27");

  // string filename = "/home/ange/WORKSPACE/AI/ml-in-action/constraint-learning/data/nsl-kdd.cnf";
  // readfile(filename);

  sat();

  return 0;
}
