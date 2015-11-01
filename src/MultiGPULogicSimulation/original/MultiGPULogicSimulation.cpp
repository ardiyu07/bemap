#include <iostream>
#include <iomanip>

#include "MultiGPULogicSimulation.hpp"

MultiGPULogicSimulation::MultiGPULogicSimulation(std::string _inputFile)
{
    /* Constructor */
    inputFile = _inputFile  ;
}

MultiGPULogicSimulation::~MultiGPULogicSimulation()
{
    /* Destructor */
}

void MultiGPULogicSimulation::init()
{
    /* Start global timer */
    t_all.start();

    t_init.start();

    //第一引数の名前のファイル名より回路データを読み込む
    module.read_BLIF(inputFile.c_str());
    
    // 全てのLFを初期化
    module.clear_all_calc_LF();

    t_init.stop();
}

void MultiGPULogicSimulation::prep_memory()
{
    t_mem.start();

    /* Initialize random number seed  */
    srand(time(NULL));

    //入力ゲートにTB割り当て   
    module.set_TB_input();

    t_mem.stop();
}

void MultiGPULogicSimulation::execute()
{
    /* Kernel Execution */
    t_kernel.start();
    
    //全ての出力ゲートのTBをCPUのみで求める
    module.calc_TBall();
    
    t_kernel.stop();
}

void MultiGPULogicSimulation::output(void *param)
{
    /* Output */

    cout << "num_inputs " << module.modulenum_inputs() << endl;
    cout << "num_outputs " << module.get_numoutputs() << endl;
    cout << "num_gates " << module.get_numgates() << endl;
    
    // For check all the gates
/*    std::ofstream out("STDOUT");
    cerr << "all the gates in the module \n";
    std::list<Gate*> gatelist = module.get_gates();
    std::list<Gate*>::iterator it_t = gatelist.begin();
    // 全てのgateをファイルへ書き出し
    while (it_t != gatelist.end() ) {
      (*it_t)->print(out);
      ++it_t;
    }*/
}

void MultiGPULogicSimulation::clean_mem()
{
    /* Cleanup */
    t_clean.start();

    t_clean.stop();
}

void MultiGPULogicSimulation::finish()
{
    /* Stop global timer */
    t_all.stop();

    /* Output time */
    showPrepTime && t_init.print_average_time("Initialization");
    showPrepTime && t_mem.print_average_time("Memory Preparation");
    t_kernel.print_average_time("Kernel : bemap_template");
    showPrepTime && t_clean.print_average_time("Cleanup");
    showPrepTime && t_all.print_total_time("Total Execution Time");
    std::cerr << std::endl;
}
