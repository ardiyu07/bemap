#include "module.hpp"

//------------------------------------------------------------------------------
//  "TB (Truth-valu)" class
//-----------------------------------------------------------------------------

int TB::TB_num_inputs; //変数TB_num_inputsの実体

TB::TB() 
{
    assert(TB_num_inputs>0);
    
    if (TB_num_inputs <= 5) {
        truth_table = new int[1];
    } 

    else if (TB_num_inputs >= 6 && TB_num_inputs <= NUM_VARIABLE) {
        truth_table = new int[1<<(TB_num_inputs-5)];
    } 

    else if (TB_num_inputs > NUM_VARIABLE) {
        truth_table = new int[1<<(NUM_VARIABLE-5)];
    }
}

//入力ゲートXiの真理値表を割り当て
TB::TB(int i) 
{
    assert(TB_num_inputs>0);
    assert(i>=0);
    
    //5変数以下ならば、配列1個
    if (TB_num_inputs <= 5) {    
        
        truth_table = new int[1];
        
        if (i==0) {
            truth_table[0] = 0x55555555;
        } else if (i==1) {
            truth_table[0] = 0x33333333;
        } else if (i==2) {
            truth_table[0] = 0x0F0F0F0F;
        } else if (i==3) {
            truth_table[0] = 0x00FF00FF;
        } else if (i==4) {
            truth_table[0] = 0x0000FFFF;
        }    
    } 

    //6変数以上、NUM_VARIABLE変数以下ならば、1<<(TB_num_inputs-5)個の配列を用意
    else if (TB_num_inputs >= 6 && TB_num_inputs <= NUM_VARIABLE) {
    
        truth_table = new int[1<<(TB_num_inputs-5)];
        
        if (i>=5) {
            for (int j=0; j < 1<<(TB_num_inputs-5); ) {
                for (int k=0; k<(1<<(i-5)); k++) {
                    truth_table[j] = 0x00000000;
                    j++;
                }
                for(int k=0; k<(1<<(i-5)); k++) {
                    truth_table[j] = 0xFFFFFFFF;
                    j++;
                }
            }
        } 
        else if (i==0) {
            for (int j=0; j < 1<<(TB_num_inputs-5); j++) {
                truth_table[j] = 0x55555555;
            }
        } 
        
        else if (i==1) {
            for (int j=0; j < 1<<(TB_num_inputs-5); j++) {
                truth_table[j] = 0x33333333;
            }
        } 
        
        else if (i==2) {
            for (int j=0; j < 1<<(TB_num_inputs-5); j++) {
                truth_table[j] = 0x0F0F0F0F;
            }
        } 
        
        else if (i==3) {
            for (int j=0; j < 1<<(TB_num_inputs-5); j++) {
                truth_table[j] = 0x00FF00FF;
            }
        } 
        
        else if (i==4) {
            for (int j=0; j < 1<<(TB_num_inputs-5); j++) {
                truth_table[j] = 0x0000FFFF;
            }
        }
    } 
    
    //NUM_VARIABLE変数以上ならば、1<<(NUM_VARIABLE-5)個の配列を用意
    else if (TB_num_inputs > NUM_VARIABLE) {
        
        truth_table = new int[1<<(NUM_VARIABLE-5)];
        
        if (i>=5) {
            for (int j=0; j < 1<<(NUM_VARIABLE-5); ) {          
                if ( i>= 36) {    //iが36の場合、左31ビットシフトするとオーバーフローするので、MAXの35とする。
                    i = 35;
                }       
                for (int k=0; k < (1<<(i-5)); k++) {    
                    if (j >= 1<<(NUM_VARIABLE-5) ) {
                        break;
                    }
                    truth_table[j] = 0x00000000;
                    j++;  
                }
                for(int k=0; k < (1<<(i-5)); k++) {
                    if (j >= 1<<(NUM_VARIABLE-5) ) {
                        break;
                    }
                    truth_table[j] = 0xFFFFFFFF;
                    j++;
                } 
            }    
        } 
        else if (i==0) {
            for (int j=0; j < 1<<(NUM_VARIABLE-5); j++) {
                truth_table[j] = 0x55555555;
            }
        } 
        else if (i==1) {
            for (int j=0; j < 1<<(NUM_VARIABLE-5); j++) {
                truth_table[j] = 0x33333333;
            }
        } 
        else if (i==2) {
            for (int j=0; j < 1<<(NUM_VARIABLE-5); j++) {
                truth_table[j] = 0x0F0F0F0F;
            }
        } 
        else if (i==3) {
            for (int j=0; j < 1<<(NUM_VARIABLE-5); j++) {
                truth_table[j] = 0x00FF00FF;
            }
        } 
        else if (i==4) {
            for (int j=0; j < 1<<(NUM_VARIABLE-5); j++) {
                truth_table[j] = 0x0000FFFF;
            }
        } 
    }
}

//TBのコピーコンストラクタ
TB::TB(const TB& from) 
{
    //  cout << "!!start copy constructerm" <<endl;
    if (TB_num_inputs <=5) {
        
        truth_table = new int[1];
        //cout << "truth_table "; this->print();
        //cout << "from_table ";
        truth_table[0] = from.truth_table[0];
        //cout << "truth_table ";this->print();
    } 
    else if (TB_num_inputs >= 6 && TB_num_inputs <= NUM_VARIABLE) {
        truth_table = new int[1<<(TB_num_inputs-5)];
        for (int i=0; i < 1<<(TB_num_inputs-5); i++) {
        truth_table[i] = from.truth_table[i];
        } 
    } 
    else if (TB_num_inputs > NUM_VARIABLE) {
        truth_table = new int[1<<(NUM_VARIABLE-5)];
        for (int i=0; i < 1<<(NUM_VARIABLE-5); i++) {
        truth_table[i] = from.truth_table[i];
        } 
    }
}

/** \fn TB:: virtual ~TB()
 *  \brief デストラクタ
 * 
 *  TBのデストラクタ
 */
TB::~TB()
{
    //登録TBを削除
    //cout << "             now deleting this          " << this << endl;
    delete [] truth_table;
}

bool TB::operator==(const TB& other) 
{
    TB tmp;
    //cout << "      +tmp  this " << &tmp << endl;
  
    if (TB_num_inputs <=5) {
        if (truth_table[0] != other.truth_table[0]) {
            cerr << "error: Not match TB" << endl;
        }
    } 
    else if (TB_num_inputs >= 6 && TB_num_inputs <= NUM_VARIABLE) {
        for (int i=0; i < 1<<(TB_num_inputs-5); i++) {
            if (truth_table[i] != other.truth_table[i]) {
                // cerr << "error: Not match TB " << i << endl;
                cerr <<  hex << setw(8) << truth_table[i] << endl;
                cerr <<  hex << setw(8) << other.truth_table[i] << endl;
                if(i==4) {
                    return(1);
                }
            } else {
                //cerr <<  hex << setw(8) << truth_table[i] << endl;
                //cerr <<  hex << setw(8) << other.truth_table[i] << endl;
                //if(i==4) {
                //return(0);
                //}
            } 
        } 
    } 
    else if (TB_num_inputs > NUM_VARIABLE) {
        for (int i=0; i < 1<<(NUM_VARIABLE-5); i++) {
            if (truth_table[i] != other.truth_table[i]) {
                // cerr << "error: Not match TB " << i << endl;
                // cerr << truth_table[i] << endl;
                // cerr << other.truth_table[i] << endl;
                return(1);
            }
        } 
    }  
    return (0);
}

TB TB::operator+(const TB& other) 
{
    TB tmp;
    //cout << "      +tmp  this " << &tmp << endl;
  
    if (TB_num_inputs <=5) {
        tmp.truth_table[0] = truth_table[0] | other.truth_table[0];
    } 
    else if (TB_num_inputs >= 6 && TB_num_inputs <= NUM_VARIABLE) {
        for (int i=0; i < 1<<(TB_num_inputs-5); i++) {
            tmp.truth_table[i] = truth_table[i] | other.truth_table[i];
        } 
    } 
    else if (TB_num_inputs > NUM_VARIABLE) {
        for (int i=0; i < 1<<(NUM_VARIABLE-5); i++) {
            tmp.truth_table[i] = truth_table[i] | other.truth_table[i];
        } 
    }
    
    eturn (tmp);
}

TB TB::operator*(const TB& other) 
{
    TB tmp;
    //cout << "      *tmp  this " << &tmp << endl;
    
    if (TBnum_inputs <=5) {
        tmp.truth_table[0] = truth_table[0] & other.truth_table[0];
    } 
    else if (TBnum_inputs >= 6 && TBnum_inputs <= NUM_VARIABLE) {
        for (int i=0; i < 1<<(TBnum_inputs-5); i++) {
            tmp.truth_table[i] = truth_table[i] & other.truth_table[i];
        } 
    } 
    else if (TBnum_inputs > NUM_VARIABLE) {
        for (int i=0; i < 1<<(NUM_VARIABLE-5); i++) {
            tmp.truth_table[i] = truth_table[i] & other.truth_table[i];
        } 
    }
    
    return (tmp);
}

TB TB::operator!() 
{
    if (TBnum_inputs <= 5) {
        truth_table[0] = ~truth_table[0];
    } 
    else if (TBnum_inputs >= 6 && TBnum_inputs <= NUM_VARIABLE) {
        for (int i=0; i < 1<<(TBnum_inputs-5); i++) {
            truth_table[i] = ~truth_table[i];
        } 
    } 
    else if (TBnum_inputs > NUM_VARIABLE) {
        for (int i=0; i < 1<<(NUM_VARIABLE-5); i++) {
            truth_table[i] = ~truth_table[i];
        }
    }
    return(*this);
}

TB TB::operator=(const TB& other) 
{
    if (this != &other) {
        
        delete[] truth_table;
        
        if (TBnum_inputs <= 5) {
            truth_table = new int[1];
            truth_table[0] = other.truth_table[0];
        } 
        
        else if (TBnum_inputs >= 6 && TBnum_inputs <= NUM_VARIABLE) {
            truth_table = new int[1<<(TBnum_inputs-5)];
            for (int i=0; i < 1<<(TBnum_inputs-5); i++) {
                truth_table[i] = other.truth_table[i];
            } 
        
        } 
        
        else if (TBnum_inputs > NUM_VARIABLE) {
            truth_table = new int[1<<(NUM_VARIABLE-5)];
            for (int i=0; i < 1<<(NUM_VARIABLE-5); i++) {
                truth_table[i] = other.truth_table[i];
            }
        }   
    }
    
    return(*this);
} 

void TB::print() 
{
    if (TB_num_inputs <= 5) {
        cout << std::hex << setw(8) << truth_table[0] << endl;
    } 

    else if (TB_num_inputs >= 6 && TB_num_inputs <= NUM_VARIABLE) {
        for (int i=0; i < 1<<(TB_num_inputs-5); i++) {
            cout << std::hex << setw(8) << truth_table[i] << "||";
            if (i == 3) break;
        }
        cout << "\n";      
    } 

    else if (TB_num_inputs > NUM_VARIABLE) {
        for (int i=0; i < 1<<(NUM_VARIABLE-5); i++) {
            cout << std::hex << setw(8) << truth_table[i] << "||";
        } 
    }
}

//-----------------------------------------------------------------------------------
//   "Gate" class
//-----------------------------------------------------------------------------------

int Gate::calc_olevel() 
{
    /*  モジュールの出力ゲートから入力ゲートに向って、それぞれのゲートにレベルをつける。
     *  レベルは、入力側は大きい、出力側は若い。 
     *  レベライズの途中で、元のレベルよりも設定したいレベルが大きい場合は、大きいレベルに更新する
     */
    int result = 0;
    if(this->get_olevel() != 0) {
        result = this->get_olevel();
    }
    //else if(this->GetType() == OUTPUT) result = 0; 
    else{
        std::list<Gate*>::iterator iter = this->output_gates.begin();
        for ( ; iter != output_gates.end() ; iter++) {
            result = max(result, (*iter)->calc_olevel()+1); 
        }
        this->set_olevel(result);
    }
    return (result);
}

int Gate::calc_ilevel() 
{
    /*
    モジュールの出力側からゲートのilevelを計算しています。
    そのために、どんどんモジュールの入力側のゲートのilevelを再帰的に計算して、
    最終的に、求めたいゲートのilevelを入力側のゲートのilevel1を参照しつつ、計算する感じにしてます（たぶん）。
    */
    int result = 0;
    
    if(this->get_ilevel() != 0) {
        result = this->get_ilevel();
    }
    else if(this->GetType() == INPUT) {
        result = 0;
    }
    else{
        std::vector<Gate*>::iterator iter = this->input_gates.begin();
        for( ; iter != input_gates.end(); iter++){
            result = max(result, (*iter)->calc_ilevel()+1);
        }
        this->set_ilevel(result);
    }
    return(result);
}

int Gate::calc_TBLF(Module* module)   //。直前のゲートのTBLFはあらかじめ計算しておく必要がある。
{
    //cout << "calcTB "<< this->name <<endl;

    if (calc_LF == 1) {
        return(0);   //計算済み
    } 

    int i, num;
    TB *f = new TB();
    //  cout << "      f  this " << &f << endl;
    TB *input[input_gates.size()];
    string gatename;

    //入力のTBを取得
    std::vector<Gate*>::iterator iter = this->input_gates.begin();
    for (i = 0; iter != input_gates.end(); iter++, i++) {
        Gate *parent_gate = *iter;

        if (parent_gate->calc_LF == true) {
            input[i] = module->find_TB(parent_gate);
            // cout <<"(*input[i]) teruth_table  parentTRUE" << i << endl; (*input[i]).print();
            if (&input[i] == NULL) {
                cerr << "error parent_gate not TB" << endl;
                exit(1);
            }
        } else {                   //未計算なので、入力ゲートのTBLFを計算
            parent_gate->calc_TBLF(module);
            input[i] = module->find_TB(parent_gate);
            // cout <<"(*input[i]) teruth_table  " << i << endl; (*input[i]).print();
        }
    }
  
    num = i;
    //TBの演算
    (*f) = (*input[0]);
    //cout <<"f truth_table"; f.print();
    //cout <<"(*input[0]) teruth_table"; (*input[0]).print();
  
    if (this->Type == NOT) {
        if (input_gates.size() != 1) {
            cerr << "error: NOTgate can't multiINPUT" << endl;
            exit(1);
        }
        (*f) = !(*f);
    }

    if (this->Type == AND) {
        for (i=1 ; i<num; i++) {
            //cout <<"before f truth_table"; (*f).print();
            //cout <<"before input teruth_table"; (*input[i]).print();
            (*f) = (*f) * (*input[i]);
            //cout <<"after f truth_table"; (*f).print();
        }
    }

    if (this->Type == OR) {
        for (i=1 ; i<num; i++) {
            //     cout <<"before f teruth_table"; (*f).print();
            // cout <<"before input teruth_table"; (*input[i]).print();
            (*f) = (*f) + (*input[i]);
            //cout <<"after f teruth_table"; (*f).print();
        }
    }

    set_calc_LF(true); //計算済フラグをたてる
    module->insert_gate2TB(this, f); //計算結果TBを挿入
    module->TBs_.push_back(f);
    //cout << "calc comp "<< this->name <<endl;
  
    return(0);   //正常に計算できた
}

/** \fn Gate::print_in_out()
 *  \brief 内容を表示する(自分と、自分の入力ゲートと出力ゲートを出力する)関数
 *  表示例
 *  gate_name = xor1
 *  gate_type = XOR
 *  input_gate = i0
 *  input_gate = i1
 *  output_gate = xor3
 *  output_gate = and4
 */ 
void Gate::print_in_out()
{
    out << "gate_name = " << this -> get_name() << endl;
    
    if( this -> get_type() == 0 ) out << "gate_type = " << "INPUT"<< endl;
    else if ( this -> get_type() == 1 ) out << "gate_type = " << "OUTPUT"<< endl;
    else if ( this -> get_type() == 2 ) out << "gate_type = " << "NAND"<< endl;
    else if ( this -> get_type() == 3 ) out << "gate_type = " << "NOR"<< endl;
    else if ( this -> get_type() == 4 ) out << "gate_type = " << "NOT"<< endl;
    else if ( this -> get_type() == 5 ) out << "gate_type = " << "AND"<< endl;
    else if ( this -> get_type() == 6 ) out << "gate_type = " << "OR"<< endl; 
    else if ( this -> get_type() == 7 ) out << "gate_type = " << "XOR"<< endl; 
    else if ( this -> get_type() == 8 ) out << "gate_type = " << "XNOR"<< endl;  
    else if ( this -> get_type() == 9 ) out << "gate_type = " << "OTHERGATES"<< endl;

    if(this->input_gates.size() !=0){
        vector<Gate*>::iterator it_in1 = this->input_gates.begin();
        while (it_in1 != this->input_gates.end() )
        {
            out << "input_gate = " << (*it_in1) -> get_name() << endl;
            ++it_in1;
        }
    }

    if(this->output_gates.size() !=0){
        list<Gate*>::iterator it_out = this->output_gates.begin();
        while (it_out != this->output_gates.end() )
        {
            out << "output_gate = " << (*it_out) -> get_name() << endl;
            ++it_out;
        }
    }
    
    out << endl;
}

/** \fn void print(std::ostream& os) 
 *  \brief 内容を表示する    ★incomplete★
 *  \brief 出力形式: gatename gatetype input1_name input2_name 
 *  \param os [in] 出力ストリーム
 */
void Gate::print(std::ostream& os)
{
    os<< this->gate_name << " " << "ilevel:" << this->ilevel << " " ; 
    
    if(this->gate_type == NAND)             os<< "NAND "; 
    else if(this->gate_type == OUTPUT)      os<< "OUTPUT "; 
    else if(this->gate_type == AND)         os<< "AND ";
    else if(this->gate_type == OR)          os<< "OR ";
    else if(this->gate_type == INPUT)       os<< "INPUT ";
    else if(this->gate_type == NOT)         os<< "NOT ";
    else if(this->gate_type == OTHERGATES)  os<< "OTHERGATES ";
    else {
        cerr << "In Gate::Print, currenlty only NAND can be treated\n";
        cout << "check TYPE " <<  this->gate_name << "gate_type: "<< this->gate_type <<endl;
        exit(1);
    }
    
    // cerr << this->input_gates.size()  << "***";
    if(this->input_gates.size() !=0)
    { 
        std::vector<Gate*>::iterator it_t = this->input_gates.begin();
        while (it_t != this->input_gates.end() )
        {
            (*it_t)->print_name(os);
            os <<" ";
            ++it_t;
        }
    };
    os << "\n";    
}

//-----------------------------------------------------------------------------------
//   "Module" class
//-----------------------------------------------------------------------------------

/** \fn Module::~Module()
 *  \brief デストラクタ
 * 
 *  Moduleのデストラクタ
 */
Module::~Module()
{
    //登録ゲートを全て削除
    std::list<Gate*>::iterator iter0 = gates_.begin();
    while (iter0 != gates_.end() ) {
        delete (*iter0);
        iter0++;
    }
}

/** \fn void Module::read_verilog( char * filename );
 *  \brief 指定されたファイルを読み込む
 *  \param filename [in] 読み込まれるファイルのファイル名
 * 
 *  filenameで指定されたファイルを読み込む
 */
void Module::read_verilog( char * filename )
{
    ifstream c_circuit_data; //ファイルの読み込み用
    
    c_circuit_data.open( filename, std::ios::in ); //ファイルの読み込みを許可
    if( !c_circuit_data ) {  //ファイルの読み込みが出来なかった場合のエラー処理
        cout << "ERROR::cannot open " << filename <<endl;
        exit(1);
    }
    string oneline;//string型のoneline変数を作成
    
    // モジュール名の処理
    while(getline(c_circuit_data, oneline) ) {  //これでonelineに一行読み込む
        
        if(oneline == "") continue; //空行を無視
        string oneword;
        oneword = get_word(oneline); //"module"という単語を飛ばすので2回get_wordを行う
        oneword = get_word(oneline); //スペースを無視して最初のモジュール名をゲット
        if (oneword.at(0) != '#' ){ //＃以降はその行を無視
            description_ = oneword;
            out << "module_name = " << oneword << endl << endl;
            break;
        }
    }
    
    //入力の処理
    while(getline(c_circuit_data, oneline) ) {  //これでonelineに一行読み込む
            
            if(oneline == "") continue; //空行を無視
            string oneword = get_word(oneline);  //スペースを無視して最初のwordをゲット

            if (oneword.at(0) != '#' ){ //＃以降はその行を無視    
                string _gate_type = oneword; //最初のonewordはゲートタイプとして受け取る（velirogの書式に依存）
                
                if( _gate_type == "input" ){
                    while( 1 ){
                        string _gate_name = get_word(oneline);
                        if(_gate_name[0] == ';') break; //';'でverilog1行の処理終了
                        else if(_gate_name[0] == ',') continue; //','を無視
                        else if(_gate_name[0] == '(') continue; //'('を無視
                        else if(_gate_name[0] == ')') continue; //')'を無視
                        
                        //ゲートの登録処理
                        Gate * newgate = new Gate(_gate_name, INPUT);
                        this->insert_name2gate(_gate_name, newgate);
                        this->gates_.push_back(newgate);
                        this->inputs_.push_back(newgate);
                    }
                }           
            }           
            break;
    }
    
    //出力の処理
    while(getline(c_circuit_data, oneline) ) {  //これでonelineに一行読み込む
            
            if(oneline == "") continue; //空行を無視
            string oneword = get_word(oneline);  //スペースを無視して最初のwordをゲット

            if (oneword.at(0) != '#' ){ //＃以降はその行を無視
                string _gate_type = oneword; //最初のonewordはゲートタイプとして受け取る（velirogの書式に依存）
                
                if( _gate_type == "output" ){
                    while( 1 ){
                        string _gate_name = get_word(oneline);
                        if(_gate_name[0] == ';') break; //';'でverilog1行の処理終了
                        else if(_gate_name[0] == ',') continue; //','を無視
                        else if(_gate_name[0] == '(') continue; //'('を無視
                        else if(_gate_name[0] == ')') continue; //')'を無視
                        
                        //ゲートの登録処理
                        Gate * newgate = new Gate(_gate_name, OUTPUT);
                        this->insert_name2gate(_gate_name, newgate); 
                        this->gates_.push_back(newgate);
                        this->outputs_.push_back(newgate);
                    }
                }
            }
            
            break;
    }

    //各ゲートの処理 (出力ゲートをつなぐのも含む）
    while(getline(c_circuit_data, oneline) ) {  //これでonelineに一行読み込む
            
            if(oneline == "") continue; //空行を無視
            string oneword = get_word(oneline);  //スペースを無視して最初のwordをゲット
            
            if( oneword == "endmodule" ) break;  //velirogの処理：endmoduleで終了
            string _gate_type = oneword;  //最初のonewordはゲートタイプとして受け取る（velirogの書式に依存）
            
            if (oneword.at(0) != '#' ){ //＃以降はその行を無視
                
                Gate *newgate;      //今回作成する新しいゲート（自ゲート）
                Gate *outputgate;   //今回作成する新しいゲートの出力ゲート
                Gate *input1gate;   //今回作成する新しいゲートの入力ゲート１
                Gate *input2gate;   //今回作成する新しいゲートの入力ゲート２
                int count = 0;
                
                while( 1 ) {         
                    
                    string _gate_name = get_word(oneline);  //次のonewordはゲート名として受け取る（velirogの書式に依存）
                    
                    if(_gate_name[0] == ';') break; //';'でverilog1行の処理終了
                    else if(_gate_name[0] == ',') continue; //','を無視
                    else if(_gate_name[0] == '(') continue; //'('を無視
                    else if(_gate_name[0] == ')') continue; //')'を無視
                    
                    //自ゲートの処理（処理の順番はvelirogの書式に依存）
                    if( count == 0 ){
                        newgate = this->set_gate( _gate_name , _gate_type );
                        count = 1;
                    }
                    
                    //出力ゲートの処理の処理（処理の順番はvelirogの書式に依存）
                    else if( count == 1 ){
                        outputgate = this->find_name(_gate_name);
                        if(outputgate == NULL){
                            outputgate = new Gate(_gate_name, OTHERGATES);
                            this->insert_name2gate(_gate_name, outputgate);
                            this->gates_.push_back(outputgate);                
                        }
                        count = 2;
                    }
                    
                    //入力ゲート１の処理の処理（処理の順番はvelirogの書式に依存）
                    else if( count == 2 ){
                        input1gate = this->find_name(_gate_name);
                        if(input1gate == NULL){
                            input1gate = new Gate(_gate_name, OTHERGATES);
                            this->insert_name2gate(_gate_name, input1gate);
                            this->gates_.push_back(input1gate);                
                        }
                        count = 3;
                    }
                    
                    //入力ゲート２の処理の処理（処理の順番はvelirogの書式に依存）
                    else if( count == 3 ){
                        input2gate = this->find_name(_gate_name);
                        if(input2gate == NULL){
                            input2gate = new Gate(_gate_name, OTHERGATES);
                            this->insert_name2gate(_gate_name, input2gate);
                            this->gates_.push_back(input2gate);
                        }
                    }
                }
                
                //自ゲートと、２つの入力ゲート、１つの出力ゲートをつなぐ
                outputgate -> add_input(newgate);
                newgate -> add_output(outputgate);
                input1gate -> add_output(newgate);
                newgate -> add_input(input1gate);
                input2gate -> add_output(newgate);
                newgate -> add_input(input2gate);
            }
    }
    
    list<Gate*>::iterator it = gates_.begin();
    while( it != gates_.end() ) {
        (*it) -> print_in_out();
        ++it;  // イテレータを１つ進める
    }
}

/** \fn void Module::read_BLIF( char * filename )
 *  \brief 指定されたBLIFファイルを読み込む
 *  \param filename [in] 読み込まれるファイルのファイル名
 * 
 *  filenameで指定されたファイルを読み込む
 */
void Module::read_BLIF( char * filename )
{
    std::ifstream c_module_data;
    c_module_data.open( filename, std::ios::in );
    
    if( !c_module_data ) {
        cerr << "ERROR::cannot open " << filename <<endl;
        exit(1);
    }

    string oneline;
     
    // モジュール名の処理
    while(getline(c_module_data, oneline) ) {  //これでonelineに一行読み込む
        
        if(oneline == "") continue; //空行を無視
        
        string oneword = get_word(oneline);  //スペースを無視して最初のwordをゲット
        
        if (oneword == ".model") {
            oneword = get_word(oneline);
        }
        //   cerr << "oneword = " << oneword;
        
        if (oneword.at(0) != '#' ) { //＃以降はその行を無視
            description_ = oneword;
            cerr << ".model = " << oneword <<endl;
            break;
        }

    }
    
    //入力の処理
    while(getline(c_module_data, oneline) ) {  //これでonelineに一行読み込む
        
        int i = 0;
        this->inputs_.resize(MAX_INPUTS);
    
        if(oneline == "") continue; //空行を無視
        
        string oneword = get_word(oneline);  //スペースを無視して最初のwordをゲット
        
        if (oneword == ".inputs" ) { //入力ゲートならば
            
            oneword = get_word(oneline);
            
            while (oneword != "") {
                Gate * g = new Gate(oneword, INPUT);
                this->insert_name2gate(oneword, g);
                this->inputs_[i] = g;
                this->gates_.push_back(g);
                oneword = get_word(oneline);
                i++;
            }
            
            this->inputs_.resize(i);
            this->num_inputs = i;
            break;
        }
    }

    //出力の処理
    while(getline(c_module_data, oneline) ) {  //これでonelineに一行読み込む
        
        if(oneline == "") continue; //空行を無視
        string oneword = get_word(oneline);  //スペースを無視して最初のwordをゲット
        
        if (oneword == ".outputs" ) { //出力ゲートならば          
            oneword = get_word(oneline);
            
            while (oneword != "") {
                Gate * newgate = this->find_name(oneword);
                
                if(newgate == NULL){ //   ゲートが無い場合、新しいゲートをとりあえず作って登録
                    newgate = new Gate(oneword, OUTPUT);
                    this->insert_name2gate(oneword, newgate);
                    this->outputs_.push_back(newgate);
                    this->gates_.push_back(newgate);
                } else {  //ゲートがある場合（入力ゲートと出力ゲートが直結の場合）：出力ゲートのリストにだけ登録
                    this->outputs_.push_back(newgate);
                }
                
                oneword = get_word(oneline);
            }
            break;
        }
    }

    
    // .namesの処理
    while(getline(c_module_data, oneline) ) {  //これでonelineに一行読み込む
        
        if (oneline == "") continue; //空行を無視
        string gatename = get_word(oneline);  //スペースを無視して最初のwordをゲット
    
        if (gatename == ".end") {
            break;
        }
        
        else if (gatename == ".names" ) { //.namesならば
    
        names:
            gatename = get_word(oneline);
            int num_input = 0; //.namesの、入力ゲート数(出力ゲートも含んでいる)
            string input_name[MAX_INPUTS]; //.namesの入力ゲート名、格納用

            //.names以降に並ぶ、入力ゲートと出力ゲートの名前をinput_name[num_input]に格納
            while (gatename != "") {
                Gate * newgate = this->find_name(gatename);
                
                if(newgate == NULL){ //   ゲートが無い場合、新しいゲートをとりあえず作って登録
                    newgate = new Gate(gatename, OTHERGATES);
                    this->insert_name2gate(gatename, newgate);
                    this->gates_.push_back(newgate);
                }
                input_name[num_input] = gatename;
                num_input++;
                gatename = get_word(oneline);
            }

            /*
             * PLA形式読込を開始
             * 
             * 以下おおまかな手順（PLAが1行だけの場合、0が出てくる場合は、細かなゲートの生成と接続を行っている）
             * 1. 1行の入力側だけを詠み込み、buf_andへ格納
             * 2. 1行の出力側だけを読み込み、入力（buf_andのゲート）をANDゲートでまとめ、ANDゲートをbuf_orに格納
             * 3. PLA形式が終るまで手順1~2を繰り返す
             * 4. 最後に、buf_orにあるゲートをORゲートでまとめる
             */
            std::vector<Gate*> buf_or;    //buf_and[]からANDをしたゲートの集まり
            int n_appear = 0; //PLAの出力部分がnegative
            int p_appear = 0; //PLAの出力部分がpositive
            int plaline = 0; //PLAの現在の行数

            while(getline(c_module_data, oneline) ) {
                
                std::vector<Gate*> buf_and;   //PLA1行に出てくるゲートの集まり
                
                if (oneline == "") continue; //空行を無視
                
                string truth_value = get_word(oneline);
                
                if (truth_value[0] == '0' || truth_value[0] == '1' || truth_value[0] == '-' ) { //PLA形式を読み込む
                    
                    //////////PLAの入力側1行を処理//////////
                    for (int i=0; i<num_input - 1; i++) {
                        
                        if (truth_value[i] == '0') {    //0の場合、Notゲートを繋ぎ、Notゲートをbuf_andに格納  
                            Gate * notgate = this->find_name("Not"+input_name[i]);
                            
                            if (notgate == NULL) { //ゲートが無い場合、新しいゲートを登録
                                notgate = new Gate("Not"+input_name[i], NOT);
                                this->insert_name2gate("Not"+input_name[i], notgate);
                                this->gates_.push_back(notgate);
                            }
                            
                            Gate * inputgate = this->find_name(input_name[i]);
                            this->connect(inputgate, notgate, 0);
                            buf_and.push_back(notgate);
                        } 
                        else if (truth_value[i] == '1') {    //1の場合、入力ゲートをbuf_andに格納
                            Gate * push = this->find_name(input_name[i]);
                            buf_and.push_back(push);
                        }
                    }
                    //////////PLAの出力側1行を処理//////////
                    
                    truth_value = get_word(oneline);

                    char str[MAX_INPUTS];          //PLAの行数を文字列へ変換
                    sprintf(str, "%d", plaline);
                    string line;
                    line = string(str);

                    int buf_and_len = buf_and.size(); //PLA1行の入力数

                    if (truth_value == "1") {  //PLA1行に相当するANDゲートを生成し、ゲート名も適切に付ける
                        p_appear = 1;
                            
                        if(buf_and_len == 1 ) {   //PLA1行の入力ゲートが1個しか無い場合,
                            //buf_orに入力ゲートをそのまま入れる
                            Gate * gate = buf_and.front();
                            buf_or.push_back(gate);
                        } else {
                            Gate * andgate = new Gate(input_name[num_input-1]+"BufAND"+line ,AND);
                            this->insert_name2gate(input_name[num_input-1]+"BufAND"+line, andgate);
                            this->gates_.push_back(andgate);
                            std::vector<Gate*>::iterator iter = buf_and.begin();
                            for (int i=0; iter != buf_and.end(); i++, ++iter) {
                                this->connect(*iter ,andgate, i);
                            }
                            buf_or.push_back(andgate);
                        }
                    }  
                        
                    else if (truth_value == "0") {    //PLA1行に相当するANDゲートを生成し、NOTゲートを後ろに接続
                        n_appear = 1;
                            
                        if(buf_and_len == 1 ) {   //PLA1行の入力ゲートが1個しか無い場合,
                            
                            //NOTゲートを作成し、buf_orに入れる
                            Gate *buf_not = this->find_name(input_name[num_input-1]+"BufNot");
                            if (buf_not == NULL) {
                                buf_not = new Gate(input_name[num_input-1]+"BufNot", NOT);
                                this->insert_name2gate(input_name[num_input-1]+"BufNot", buf_not);
                                this->gates_.push_back(buf_not);           
                            }
                            
                            this->connect(buf_and.front(), buf_not, 0);
                            buf_or.push_back(buf_not);
                        } else {      
                            
                            Gate * andgate = new Gate(input_name[num_input-1]+"BufAND"+line ,AND);
                            this->insert_name2gate(input_name[num_input-1]+"BufAND"+line, andgate);
                            this->gates_.push_back(andgate);
                            std::vector<Gate*>::iterator iter = buf_and.begin();
                                
                            for (int i=0; iter != buf_and.end(); i++, ++iter) {
                                this->connect(*iter ,andgate, i);
                            }
                                
                            Gate * buf_not = new Gate(input_name[num_input-1]+"BufAND"+"Not" ,NOT);
                            this->insert_name2gate(input_name[num_input-1]+"BufAND"+"Not", buf_not);
                            this->gates_.push_back(buf_not); 
                            this->connect(andgate ,buf_not ,0);
                            buf_or.push_back(buf_not);
                        }
                    }
        
                    plaline++;
                    /*
                     * PLA形式読込を終了
                     *
                     */
                } 

                else if (truth_value == ".names") {  //ゲートを生成し、.namesへジャンプ
        
                    int buf_or_len = buf_or.size(); //PLAの行数 

                    if (p_appear && n_appear) {
                        cerr << "PLAformat error " << "before: .names " << oneline << endl;
                        exit(1);
                    } 

                    else if (p_appear) {
                          
                        if (buf_or_len == 1 ) {     //PLAが１行しかないので,buf_orにあるゲートを出力へ繋ぐ
                            Gate * outgate = this->find_name(input_name[num_input-1]);
                            Gate * ingate = buf_or.front();
                            this->connect(ingate, outgate, 0);
                        } else {                    //PLAが複数行の場合,buf_ofにある全てのゲートをORゲートに繋ぐ
                            Gate * orgate = this->find_name(input_name[num_input-1]);
                            std::vector<Gate*>::iterator iter = buf_or.begin();
                            for (int i=0; iter != buf_or.end(); i++, ++iter) {
                                this->connect(*iter ,orgate, i);
                            }
                            orgate->SetType(OR);
                        }
                    } 

                    else if (n_appear) {    //PLA形式の出力が0の時、PLAは1行しかない
                            
                        if (buf_or_len == 1 ) {     //PLAが１行しかないので,buf_orにあるゲートを出力へ繋ぐ
                            Gate * outgate = this->find_name(input_name[num_input-1]);
                            Gate * ingate = buf_or.front();
                            this->connect(ingate, outgate, 0);
                        } else {
                            cerr << "PLAformat error " << "before: .names " << oneline << endl;
                            cout << "not allow multiline(case: output0)"<< endl;
                            exit(1);
                        }
                    }  

                    goto names;
                } 

                else if (truth_value == ".end") {
        
                    int buf_or_len = buf_or.size(); //PLAの行数 

                    if (p_appear && n_appear) {
                        cerr << "PLAformat error " << "before: .names " << oneline << endl;
                        exit(1);
                    } 

                    else if (p_appear) {
                        if (buf_or_len == 1 ) {     //PLAが１行しかないので,buf_orにあるゲートを出力へ繋ぐ
                            Gate * outgate = this->find_name(input_name[num_input-1]);
                            Gate * ingate = buf_or.front();
                            this->connect(ingate, outgate, 0);
                        } else {                    //PLAが複数行の場合,buf_ofにある全てのゲートをORゲートに繋ぐ
                            Gate * orgate = this->find_name(input_name[num_input-1]);
                            std::vector<Gate*>::iterator iter = buf_or.begin();
                            for (int i=0; iter != buf_or.end(); i++, ++iter) {
                              this->connect(*iter ,orgate, i);
                            }
                            orgate->SetType(OR);
                        }
                    } 

                    else if (n_appear) {    //PLA形式の出力が0の時、PLAは1行しかない
                        if (buf_or_len == 1 ) {     //PLAが１行しかないので,buf_orにあるゲートを出力へ繋ぐ
                          Gate * outgate = this->find_name(input_name[num_input-1]);
                          Gate * ingate = buf_or.front();
                          this->connect(ingate, outgate, 0);
                        } else {
                          cerr << "PLAformat error " << "before: .names " << oneline << endl;
                          cout << "not allow multiline(case: output0)"<< endl;
                          exit(1);
                        }
                    }  
                }   
            }
        }
    }
}

void Module::clear_all_calc_LF()
{
    //登録ゲートのLFを全て初期化
    std::list<Gate*>::iterator iter0 = gates_.begin();
    while (iter0 != gates_.end() ) {
      Gate * gate = *iter0;
      gate->set_calc_LF(0);
      iter0++;
    }  
}

void Module::set_TB_input()
{
    TB::set_TB_numinputs(modulenum_inputs());
    std::vector<Gate*> inputs_ = get_inputs();
    std::vector<Gate*>::reverse_iterator iter = inputs_.rbegin();
    
    for (int i = 0; iter != inputs_.rend(); iter++, i++) {
        Gate * gate = *iter;
        
        //TB割り当て
        TB * tb = this->find_TB(gate);
        if (tb == NULL) { //真理値表を新規に割り当て
            tb = new TB(i);
            this->insert_gate2TB(gate, tb);
            this->TBs_.push_back(tb);
            gate->set_calc_LF(true); //計算済フラグをたてる
        } else { //大規模回路をシミュレーションする場合（真理値表を分割してセットする場合）
            // 真理値表の更新2回目3回目4回目・・・
            cerr << "unimplemented reset_TB_input, but simulation is OK" << endl;
            gate->set_calc_LF(true); //計算済フラグをたてる
        }
    }
}

/** \fn string Module::get_word(std::string &oneLine)
 *  \brief 指定された文字列の文字列の先頭からコンマまでの文字列を取得すす
 *  \param oneLine [in, out] 文字列
 *  \return 取得された文字列
 *
 *  oneLineで与えられた文字列からコンマで区切られた文字列の先頭を切り出して
 *  返す。また、oneLineは切り出された文字列が削除される。返される文字列の
 *  前後のコンマと空白文字は無視される。
 *
 *  例) oneLine = "a , b , c , f" の場合
 *  "a"が返され、oneLine = "b , c , d"となる。
 */
string Module::get_word(std::string &oneLine)
{
    if ( oneLine.size() == 0 ) {
      return "";
    }
    
    string::size_type position = oneLine.find_first_not_of(" \t");
    
    if ( position > oneLine.size() ) {
      position = oneLine.size();
    }
    oneLine.erase(0,position);
    
    position = oneLine.find_first_of(" \t");
    
    if ( position > oneLine.size() ) {
      position = oneLine.size();
    }
    string word = oneLine.substr(0, position);
    oneLine.erase(0,position+1);
    
    return word;
}

int Module::set_olevel_vector()
{
    //全てのゲートの初期化 
    std::list<Gate*>::iterator iter = gates_.begin();
    while (iter != gates_.end() ) {
        Gate *gate = *iter;
        gate->set_olevel(0);
        iter++;
    }
 
    //レベル付け
    std::vector<Gate*> input_gatelist = get_inputs();
    std::vector<Gate*>::iterator i_iter = input_gatelist.begin();
    while (i_iter != input_gatelist.end() ) {
        (*i_iter)->calc_olevel();
        ++i_iter;
    }

    //Max_levelをセットする。（全ての入力ゲートを見て、最大レベルを取得）
    set_max_level(0);
    input_gatelist = get_inputs();
    i_iter = input_gatelist.begin(); 
    while (i_iter != input_gatelist.end() ) {
        int level = (*i_iter)->get_olevel();
        int maxlevel = get_max_level();
        if (level > maxlevel) {
          set_max_level(level);
        }
        ++i_iter;
    }

    //全てのゲートをレベル毎に、コンテナのコンテナに仕分けする。
    int maxlevel = get_max_level();
    olevelVector.resize(maxlevel + 1); //0~maxlevel

    iter = gates_.begin();
    while (iter != gates_.end() ) {
        Gate * gate = *iter;
        int level = gate->get_olevel();
        olevelVector[level].push_back(gate);
        iter++;
    }
    return(0);    //レベライズ完了
}

int Module::set_ilevelVector()
{
    //全てのゲートの初期化
    std::list<Gate*>::iterator iter = gates_.begin();
    while (iter != gates_.end() ) {
        Gate *gate = *iter;
        gate->set_ilevel(0);
        iter++;
    }

    //レベル付け,各モジュールに対してilevelを設定
    std::list<Gate*> output_gatelist = get_outputs();
    std::list<Gate*>::iterator o_iter = output_gatelist.begin();
    while (o_iter != output_gatelist.end() ){
        (*o_iter)->calc_ilevel();
        ++o_iter;
    }

    //Max_levelをセットする。（全ての入力ゲートを見て、最大レベルを取得）
    set_max_level(0);

    output_gatelist = get_outputs();
    o_iter = output_gatelist.begin(); 
    while (o_iter != output_gatelist.end() ) {
        int level = (*o_iter)->get_ilevel();
        int maxlevel = get_max_level();
        if (level > maxlevel) {
            set_max_level(level);
        }
        ++o_iter;
    }

    //全てのゲートをレベル毎に、コンテナのコンテナに仕分けする。
    int maxlevel = get_max_level();
    ilevelVector.resize(maxlevel + 1); //0~maxlevel

    iter = gates_.begin();
    while (iter != gates_.end() ) {
        Gate * gate = *iter;
        int level = gate->get_ilevel();
        ilevelVector[level].push_back(gate);
        iter++;
    }

    return(0);    //レベライズ完了
}

int Module::calc_TBall()
{
    std::list<Gate*> output_gatelist = get_outputs();
    std::list<Gate*>::iterator iter = output_gatelist.begin();  
    
    // 全ての出力ゲートのTBLFを計算する
    while (iter != output_gatelist.end() ) {
        (*iter)->calc_TBLF(this);
        ++iter;
    }
    
    return(0);   //正常に計算できた
}

int Module::calc_TBall_CUDA()
{
    std::list<Gate*> output_gatelist = get_outputs();
    std::list<Gate*>::iterator iter = output_gatelist.begin();  
    
    // 全ての出力ゲートのTBLFを計算する
    while (iter != output_gatelist.end() ) {
        (*iter)->calc_TBLF_CUDA(this);
        ++iter;
    }
    
    return(0);   //正常に計算できた
}

/** \fn void connect(Gate * former_g, Gate * latter_g, int i)
 *  \brief former_gの出力をlatter_gのi番目の入力につなぐ
 *  
 *  
 */                          
void Module::connect(Gate * former_g, Gate * latter_g, int i)
{
    // .outputs(出力ゲート)が、他のゲートより前に来ること許可 以下5行コメントアウト
    // assert(former_g->GetType() != OUTPUT);
    // if(former_g->GetType() == OUTPUT) {
    //  cout << former_g ->name <<"type: " << former_g->Type << endl;
    //  exit(1);
    //}
    assert(latter_g->GetType() != INPUT);
    
    former_g->addOutput(latter_g);
    latter_g->addInput(former_g, i);
}

void Module::simulate_mask_rate()
{
    //----------------simulate_mask_rate時間測定スタート-----------------------------------------
    clock_t  start, end;
    
    start = clock();
    
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////      ↓ここから出力格納処理
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    long input_count = 0, flag_count = 0, input_gate_count = 0;
    double inputs_size = inputs_.size(); //外部入力のサイズ（ビット数）
    
    map<Gate*,bool> gate2output; //自ゲートをkey,自ゲートの出力値をvalueとしたmap
    map<Gate*,bool> gate2calc; //自ゲートをkey,valueは自ゲートの出力値が確定しているならtrue,確定していないならfalseが入るmap
    
    long max_input_value = (long)pow(2.0,inputs_size); //外部入力の最大値+1(3bitだと8)
    char input_value[inputs_.size()]; //外部入力の値を格納(6の場合１、１、０)
    
    bool **gate_output_value; //各ゲートの出力値を格納
    gate_output_value = new bool*[gates_.size()];
    
    for(int i = 0; i < gates_.size(); i++) {
        gate_output_value[i] = new bool[max_input_value];
    }
    //bool gate_output_value[gates_.size()][max_input_value]; //各ゲートの出力値を格納

    //out<<"---------------------------nonreturn------------------------------"<<endl;
    for(long x = 0; x < max_input_value; x++) { //全input_valueパターンのループスタート

        //入力を2進数にする処理（7⇒1,1,1）
        int bit = 1;
        for (long i = inputs_.size()-1; i >= 0; i--) {
            if (x & bit) input_value[i] = '1';
            else input_value[i] = '0';
            bit <<= 1;
        }

        //----マップの初期化処理------------------------------------
        
        gate2output.clear();
        gate2calc.clear();
        list<Gate*>::iterator it1 = gates_.begin();
        
        while( it1 != gates_.end() ) {
            gate2output.insert(pair<Gate*, bool>((*it1), false));
            gate2calc.insert(pair<Gate*, bool>((*it1), false));
            ++it1;
        }
        
        //----ここまでマップの初期化処理------------------------------------
        
        input_count = 0;
    
        while( 1 ) { //全gateのループスタート（ここでは各ゲートの正しい出力値を求める）

            //全ゲートが出力値の計算を終えたかのチェック
            flag_count = 0;
            list<Gate*>::iterator it_FLAG = gates_.begin();
            
            while ( it_FLAG != gates_.end() ){
                if(gate2calc[(*it_FLAG)] == true) flag_count++;
                ++it_FLAG;
            }
            
            if( flag_count == gates_.size() ) break; //全てのゲートの演算が終了した場合、ループを抜ける

            list<Gate*>::iterator IN = gates_.begin();
            while( IN != gates_.end() ) {
                
                if( gate2calc[(*IN)] == false){

                    if( (*IN) -> get_type() == 0 ){ //自ゲートが0 INPUTタイプの場合、if処理の中に入る
                        if(input_value[input_count] == '1') gate2output[(*IN)] = true; //input_countビット目が1の場合入力値にtrueを入れる
                        else if(input_value[input_count] == '0') gate2output[(*IN)] = false; //input_countビット目が0の場合入力値にfalseを入れる
                        else cout << "error not '1' and'0' input" << endl;
                        gate2calc[(*IN)] = true;
                        input_count++;
                    }
                    
                    else { //自ゲートが0 INPUTタイプでない場合、else処理の中に入る
                        input_gate_count = 0;

                        //自ゲートの前段のゲートの出力値が確定しているかのチェック
                        vector<Gate*>IPS = (*IN) -> get_input_gate();
                        vector<Gate*>::iterator it_in2 = IPS.begin();
                        
                        while (it_in2 != IPS.end()){
                            if(gate2calc[(*it_in2)] == true) {  
                                input_gate_count++; 
                            }
                            ++it_in2;
                        }

                        if( input_gate_count == IPS.size()) { //自ゲートの前段のゲートの出力値が確定しているなら、if処理の中に入る
                            
                            vector<Gate*>::iterator it_OP = IPS.begin();
                            
                            while (it_OP != IPS.end() ) {
                            
                                if(it_OP == IPS.begin() && (*IN) -> get_type() == 4) {
                                    gate2output[(*IN)] = gate2output[(*it_OP)]; //4 NOTタイプの場合
                                } else {
                                    if(it_OP == IPS.begin()) gate2output[(*IN)] = gate2output[(*it_OP)]; //1 OUTPUTタイプの場合と、その他のゲートの１入力目
                                    else if((*IN)->get_type() == 2) gate2output[(*IN)] = ~(gate2output[(*IN)] & gate2output[(*it_OP)]); //2 NANDタイプの場合
                                    else if((*IN)->get_type() == 3) gate2output[(*IN)] = ~(gate2output[(*IN)] | gate2output[(*it_OP)]); //3 NORタイプの場合
                                    else if((*IN)->get_type() == 5) gate2output[(*IN)] = gate2output[(*IN)] & gate2output[(*it_OP)]; //5 ANDタイプの場合
                                    else if((*IN)->get_type() == 6) gate2output[(*IN)] = gate2output[(*IN)] | gate2output[(*it_OP)]; //6 ORタイプの場合
                                    else if((*IN)->get_type() == 7) gate2output[(*IN)] = gate2output[(*IN)] ^ gate2output[(*it_OP)]; //7 XORタイプの場合
                                    else if((*IN)->get_type() == 8) gate2output[(*IN)] = ~(gate2output[(*IN)] ^ gate2output[(*it_OP)]); //8 XNORタイプの場合
                                    else cout << "error othergate exist" << endl;
                                }
                                ++it_OP;    
                            }
                            gate2calc[(*IN)] = true; //ゲートの演算が終わったのでgate2calcにtrueを入れる
                        }
                    }

                }
                ++IN;
            }

        } //全gateのループエンド
    
        //各ゲートの出力値を格納
        long count_gate = 0;
        list<Gate*>::iterator it2 = gates_.begin();
        while( it2 != gates_.end() ) {
            gate_output_value[count_gate][x] = gate2output[(*it2)];
            count_gate++;
            ++it2;
        }
    
    } //全input_valueパターンのループエンド
    
    //出力処理(今回は出力するだけなのでコメントアウト)
    /*
    for(long x = 0; x < max_input_value; x++){
        int count_gate = 0;
        out << "----------------------------------------" << endl;
        out << "input_value = " << x << endl << endl;
        list<Gate*>::iterator it2 = gates_.begin();
        while( it2 != gates_.end() ) {
            out << "Gate " << (*it2) -> get_name() << " out = " << gate_output_value[count_gate][x] << endl;
            ++it2;
            count_gate++;
        }
        out << "----------------------------------------" << endl;
    }
    */
    
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////      ↑ここまで出力格納処理
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////      ↓ここから反転処理
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    map<Gate*,bool> gate2returnoutput; //自ゲートをkey,ゲートの出力が反転した後の自ゲートの出力値をvalueとしたmap
    long dont_care[max_input_value]; //入力パターンiの時の論理マスク発生回数
    
    for(long DC = 0; DC < max_input_value; DC++) {
        dont_care[DC] = 0; //dont_careの初期化
    }

    long count_return_gate = 0;
    long real_gate_count = 0;
    
    list<Gate*>::iterator it_return = gates_.begin();
    
    while (it_return != gates_.end() ) { //全ゲートの反転パターンのループスタート
    
        //入力、出力、アザーゲート以外の場合のみ出力を反転するので例外処理
        if( !((*it_return) -> get_type() == 0 || (*it_return) -> get_type() == 1 || (*it_return) -> get_type() == 9 ) ){
            //out<< endl <<"------------------------------"<< (*it_return)->get_name() << "return--------------------------------" <<endl;
            real_gate_count++;
        }
    
        for(long x = 0; x < max_input_value; x++) { //全input_valueパターンのループスタート
        
            //入力を2進数にする処理（7⇒1,1,1）
            int bit = 1;
            for (long i = inputs_.size()-1; i >= 0; i--) {
                if (x & bit) input_value[i] = '1';
                else input_value[i] = '0';
                bit <<= 1;
            }

            //----マップの初期化処理------------------------------------
            gate2returnoutput.clear();
            gate2calc.clear();
            list<Gate*>::iterator it1 = gates_.begin();
            while( it1 != gates_.end() ) {
                gate2returnoutput.insert(pair<Gate*, bool>((*it1), false));
                gate2calc.insert(pair<Gate*, bool>((*it1), false));
                ++it1;
            }
            //----ここまでマップの初期化処理------------------------------------
            
            if( (*it_return) -> get_type() == 0 ) break; //ゲートが入力タイプなら反転処理を行わない
            if( (*it_return) -> get_type() == 1 ) break; //ゲートが出力タイプなら反転処理を行わない
            
            //ここでゲート出力の反転処理を行う
            if(gate_output_value[count_return_gate][x] == true) gate2returnoutput[(*it_return)] = false;
            else if(gate_output_value[count_return_gate][x] == false) gate2returnoutput[(*it_return)] = true;
            else cout << "error" << endl;
            gate2calc[(*it_return)] = true;
        
            input_count = 0;
            
            while( 1 ) { //全gateのループスタート（ここではとあるゲートの出力値が反転した状態で、各ゲートの正しい出力値を求める）
            
                //全ゲートが出力値の計算を終えたかのチェック
                flag_count = 0;
                list<Gate*>::iterator it_FLAG = gates_.begin();
                
                while (it_FLAG != gates_.end() ) {
                    if(gate2calc[(*it_FLAG)] == true) flag_count++;
                    ++it_FLAG;
                }
                
                if(flag_count == gates_.size()) break; //全てのゲートの演算が終了した場合、ループを抜ける
            
                list<Gate*>::iterator IN = gates_.begin();
            
                while( IN != gates_.end() ) {           
                    if( gate2calc[(*IN)] == false){
                
                        if( (*IN) -> get_type() == 0 ) { //自ゲートが0 INPUTタイプの場合、if処理の中に入る
                            if(input_value[input_count] == '1') gate2returnoutput[(*IN)] = true; //input_countビット目が1の場合入力値にtrueを入れる
                            else if(input_value[input_count] == '0') gate2returnoutput[(*IN)] = false; //input_countビット目が0の場合入力値にfalseを入れる
                            else cout << "error not '1' and'0' input" << endl;
                            gate2calc[(*IN)] = true;
                            input_count++;
                        }
                        else {  //自ゲートが0 INPUTタイプでない場合、else処理の中に入る
                            
                            input_gate_count = 0;
                            
                            //自ゲートの前段のゲートの出力値が確定しているかのチェック
                            vector<Gate*>IPS2 = (*IN) -> get_input_gate();
                            vector<Gate*>::iterator it_in3 = IPS2.begin();
                        
                            while (it_in3 != IPS2.end() ){
                                if(gate2calc[(*it_in3)] == true) input_gate_count++; 
                                ++it_in3;
                            }
                        
                            if( input_gate_count == IPS2.size()) { //自ゲートの前段のゲートの出力値が確定しているなら、if処理の中に入る
                                vector<Gate*>::iterator it_OP = IPS2.begin();
                                
                                while(it_OP != IPS2.end() ){
                                
                                    if(it_OP == IPS2.begin() && (*IN) -> get_type() == 4) {
                                        gate2returnoutput[(*IN)] = gate2returnoutput[(*it_OP)]; //4 NOTタイプの場合
                                    }
                                
                                    else {
                                        if(it_OP == IPS2.begin()) gate2returnoutput[(*IN)] = gate2returnoutput[(*it_OP)]; //1 OUTPUTタイプの場合と、その他のゲートの１入力目
                                        else if((*IN) -> get_type() == 2) gate2returnoutput[(*IN)] = ~(gate2returnoutput[(*IN)] & gate2returnoutput[(*it_OP)]); //2 NANDタイプの場合
                                        else if((*IN) -> get_type() == 3) gate2returnoutput[(*IN)] = ~(gate2returnoutput[(*IN)] | gate2returnoutput[(*it_OP)]); //3 NORタイプの場合
                                        else if((*IN) -> get_type() == 5) gate2returnoutput[(*IN)] = gate2returnoutput[(*IN)] & gate2returnoutput[(*it_OP)]; //5 ANDタイプの場合
                                        else if((*IN) -> get_type() == 6) gate2returnoutput[(*IN)] = gate2returnoutput[(*IN)] | gate2returnoutput[(*it_OP)]; //6 ORタイプの場合
                                        else if((*IN) -> get_type() == 7) gate2returnoutput[(*IN)] = gate2returnoutput[(*IN)] ^ gate2returnoutput[(*it_OP)]; //7 XORタイプの場合
                                        else if((*IN) -> get_type() == 8) gate2returnoutput[(*IN)] = ~(gate2returnoutput[(*IN)] ^ gate2returnoutput[(*it_OP)]); //8 XNORタイプの場合
                                        else cout << "error othergate exist" << endl;
                                    }
                                    ++it_OP;
                                }
                                
                                gate2calc[(*IN)] = true; //ゲートの演算が終わったのでgate2calcにtrueを入れる
                            }
                        } 
                    }
                    ++IN;
                }
            } //全gateのループエンド
        
            int DCcount = 0;
            //out << "----------------------------------------" << endl;
            //out << "input_value = " << x << endl << endl;
            list<Gate*>::iterator it5 = gates_.begin();
            int GateCount = 0;
        
            while( it5 != gates_.end() ) {        
                if((*it5)->get_type() == 1) {
                    //out << "Gate " << (*it5) -> get_name() << " return_out = " << gate2returnoutput[(*it5)] << endl;
                    //out << "Gate " << (*it5) -> get_name() << " out = " << gate_output_value[GateCount][x] << endl;
                    if(gate2returnoutput[(*it5)] == gate_output_value[GateCount][x]) { 
                        DCcount++; //反転する前の外部出力と反転した後の外部出力がびっと単位で一致した数を数える
                    }
                }
                ++it5;
                GateCount++;
            }
        
            //out << "----------------------------------------" << endl;

            if(DCcount == outputs_.size()) { //反転する前の外部出力と反転した後の外部出力が一致した場合の処理
                dont_care[x]++;
                //out << "When input_value = " << x << " and return_gate = " << (*it_return) -> get_name() << " happen mask " << endl;
            }
        } //全input_valueパターンのループエンド
        
        count_return_gate++;
        ++it_return;

    } //全ゲートの反転パターンのループエンド
    
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////      ↑ここまで反転処理
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    //メモリの開放
    for(int i = 0; i < outputs_.size(); i++){
        delete [] gate_output_value[i];
    }
    delete [] gate_output_value;
    
    /////////////////////////////////////ここからマスク発生率を求めていく////////////////////////////////////////////////////////  
    out << endl;
    float AMR;
    float MR;
    float MaskRate[max_input_value];
    
    for(long x = 0; x < max_input_value; x++) {
        MaskRate[x] = dont_care[x] / (float) real_gate_count;
        out << "input = " << x << " is " << MaskRate[x] << " MaskRate;" << endl;
        MR += MaskRate[x];
    }
    
    cout << MR << endl;
    cout << max_input_value << endl;
    AMR = MR / (float) max_input_value;
    out << endl << "This module's AverageMaskRate = "  << AMR << endl;

    cout << "----------Finish----------" << endl;
    /////////////////////////////////////ここまででマスク発生率を求めるのに成功////////////////////////////////////////////////////////
    
    end = clock();
    out << (double)(end-start)/CLOCKS_PER_SEC << endl;
    
    //----------------simulate_mask_rate時間測定エンド-----------------------------------------
}