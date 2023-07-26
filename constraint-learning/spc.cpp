#include <cryptominisat5/cryptominisat.h>
#include <assert.h>
#include <vector>
using std::vector;
using namespace CMSat;

int main()
{
  SATSolver solver;
  vector<Lit> clause;

  // Let's use 4 threads
  solver.set_num_threads(4);

  // We need 3 variables. They will be: 0,1,2
  // Variable numbers are always trivially increasing
  solver.new_vars(3);

  // add "1 0"
  clause.push_back(Lit(0, false));
  solver.add_clause(clause);

  // add "-2 0"
  clause.clear();
  clause.push_back(Lit(1, true));
  solver.add_clause(clause);

  // add "-1 2 3 0"
  clause.clear();
  clause.push_back(Lit(0, true));
  clause.push_back(Lit(1, false));
  clause.push_back(Lit(2, false));
  solver.add_clause(clause);

  lbool ret = solver.solve();
  assert(ret == l_True);
  std::cout
      << "Solution is: "
      << solver.get_model()[0]
      << ", " << solver.get_model()[1]
      << ", " << solver.get_model()[2]
      << std::endl;

  // assumes 3 = FALSE, no solutions left
  vector<Lit> assumptions;
  assumptions.push_back(Lit(2, true));
  ret = solver.solve(&assumptions);
  assert(ret == l_False);

  // without assumptions we still have a solution
  ret = solver.solve();
  assert(ret == l_True);

  // add "-3 0"
  // No solutions left, UNSATISFIABLE returned
  clause.clear();
  clause.push_back(Lit(2, true));
  solver.add_clause(clause);
  ret = solver.solve();
  assert(ret == l_False);

  return 0;
}