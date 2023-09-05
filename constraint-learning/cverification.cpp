#include <cryptominisat5/cryptominisat.h>
#include <assert.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string.h>
#include <vector>

#define FILENAME_MINI "test/6-minimized-cnf.cnf"
#define FILENAME_FULL "test/6-full-cnf.cnf"
#define FILENAME_EXCL "test/6-excluded-cnf.cnf"

using namespace CMSat;
using namespace std;

void print_satisfiable_sample(SATSolver const &solver)
{
  cout << "Solution is: \t" << solver.get_model()[0];
  for (int i = 1; i < solver.nVars(); i++)
    cout << ", " << solver.get_model()[i];
  cout << endl;
}

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
  vector<vector<int>> data;
  vector<int> shape;

  // Open the file
  fstream file(filename);

  if (file)
  {
    string line = "";

    /** Get shape at the 1st line of the file */
    getline(file, line);
    shape = parse_cast_and_convert_to_vect(line);

    cout << "Processing FILE \"" << filename << "\"." << endl
         << "Shape = (" << shape[1] << ", " << shape[0] << ")" << endl
         << endl;

    /** Rest of the file: Store the line as a row of the 2D vector */
    while (getline(file, line))
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

  /** Add shape at the end */
  data.push_back(shape);

  return data;
}

void sat()
{
  SATSolver solver;
  vector<Lit> clause;
  lbool ret;
  int count = 0;
  vector<vector<int>> cnf_arr = readfile(FILENAME_FULL);

  /** Shape ++ Variables size definition */
  vector<int> shape = cnf_arr[cnf_arr.size() - 1];
  int variables = shape[0],
      threads = shape[1];
  // cout << threads << " - " << variables << endl;

  /** Remove the shape at the end before processing */
  cnf_arr.pop_back();
  solver.set_num_threads(threads);
  solver.new_vars(variables);
  // return;

  /*******************************************************************************/
  for (int i = 0; i < threads; i++)
  {
    clause.clear();

    for (int j = 0; j < cnf_arr[i].size(); j++)
      clause.push_back(Lit(abs(cnf_arr[i][j]) - 1, cnf_arr[i][j] < 0));

    solver.add_clause(clause);
  }

  /*******************************************************************************/
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
    print_satisfiable_sample(solver);
    cout << "# sample |----->  " << count << endl;

    /** Banning found solution */
    vector<Lit> ban_solution;

    for (uint32_t var = 0; var < solver.nVars(); var++)
      if (solver.get_model()[var] != l_Undef)
        ban_solution.push_back(Lit(var, (solver.get_model()[var] == l_True) ? true : false));

    solver.add_clause(ban_solution);
  }
}

int main()
{
  sat();

  return 0;
}
