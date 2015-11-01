#ifndef __MULTIGPULOGICSIMULATION_MODULE_HPP
#define __MULTIGPULOGICSIMULATION_MODULE_HPP

#include <iostream>
#include <iomanip>
#include <fstream>

#include <list>
#include <vector>
#include <string>
#include <map>

#include <cassert>
#include <cmath>

#include <ctime>
#include <sys/time.h>

#define MAX_INPUTS 2096    //ゲートへの最大入力数
#define NUM_VARIABLE 25    //真理値表で扱う変数の数　警告：36以上を設定するとオーバーフロー　真理値表サイズ：（2^NUM_VARIABLE）bit
#define MAX_GATE 1250      //(NUM_VARIABLE + CASH_GATE) GPUに置ける最大の真理値表の数. GPUのメモリ量に注意.
#define CASH_GATE 1200     //GPUに置く真理値表の数　MAX_GATEよりも小さい値を設定

using namespace::std;

//ゲートのタイプ分け
typedef enum {
    INPUT,      //0 INPUTタイプ
    OUTPUT,     //1 OUTPUTタイプ
    NAND,       //2 NANDタイプ  
    NOR,        //3 NORタイプ
    NOT,        //4 NOTタイプ  
    AND,        //5 ANDタイプ  
    OR,         //6 ORタイプ  
    XOR,        //7 XORタイプ  
    XNOR,       //8 XNORタイプ 
    OTHERGATES  //9 上記以外のタイプ  
} GateType;

class Module;
class Gate;
class TB;

//------------------------------------------------------------------------------
//  "TB (Truth-valu)" class
//-----------------------------------------------------------------------------

/** \class TB
 *  \brief TB クラス
 *
 */
class TB {

private:

  static int TB_num_inputs;    //静的メンバ: 入力ゲート数

public:

  /** \var int truth_table[];
   *  \brief Gateの真理値表
   */
  int *truth_table;

  /** \fn TB()
   *  \brief コンストラクタ
   *  \param 真理値表を指定する。
   */
  TB();                  //デフォルトコンストラクタ（真理値表をnew）
  TB(int i);             //入力ゲートXiの真理値表を割り当て
  TB(const TB& from);    //コピーコンストラクタ
  
  // Operator Overload

  TB operator+(const TB& other);
  
  TB operator*(const TB& other);
  
  TB operator!();
  
  TB operator=(const TB& other);
  
  bool operator==(const TB& other);
  
  // Other Functions

  void print();    //真理値表を表示
  
  static void set_TB_numinputs(int n) {    //入力ゲート数をセット
    TB_num_inputs = n;
  }  

  /** \fn virtual ~TB()
   *  \brief デストラクタ
   */
  virtual ~TB();

};

//-----------------------------------------------------------------------------------
//   "Gate" class
//-----------------------------------------------------------------------------------

class Gate {

private:

    /** \var std::string name;
     *  \brief ゲートの名前，空でもOK（出力時は適当な名前になる）
     *  \brief INPUT -> i1, i2, ...
     *  \brief OUTPUT -> o1, o2, ...
     */

    
    /** \var GateType gate_type
     *  \brief ゲートのタイプ
     */
    GateType gate_type;
    
    /** \var int ilevel
     *  \brief 入力側から割り振ったゲートのレベル
     */
    int ilevel;

    /** \var int olevel
     *  \brief 出力側から割り振ったゲートのレベル
     */
    int olevel;

    /** \var int count_outputs
     *  \brief ゲートの出力先の数を表す
     *  - 正の数：GPU内に存在する
     *  - 0：GPU内に存在しない
     */
    int count_outputs;
    
    /** \var int flag_cuda_d
     *  \brief ゲートのTBがGPUに存在するかを示す
     *  - 1:GPU内に存在する
     *  - 0：GPU内に存在しない
     */
    int flag_cuda_d;
    

    /** \var int output_val
     *  \brief ゲートの出力値
     */
    int output_val;
    
    /** \var std::vector<Gate*> input_gates
     *  \brief 自分の入力がつながっているgateの配列
     *  \brief ★ゲートによっては順番に意味がある★
     */
    vector<Gate*> input_gates;
    
    /** \var std::list<Gate*> output_gates
     *  \brief 自分の出力がつながっているgateの配列
     */
    list<Gate*> output_gates;
    
    /** \var bool calc_LF
     *  \brief LFが計算されているかどうか
     *
     *  - true 計算済み
     *  - false 未計算
     */
    bool calc_LF;

public:

    string gate_name;

    /** \fn Gate()
     *  \brief コンストラクタ
     *  \param gate_type [in] ビットのタイプを指定する。OTHERGATESがデフォルト
     */

    Gate(string _gate_name, GateType _gate_type = OTHERGATES)
    {
        gate_name = _gate_name;
        gate_type = _gate_type;
    };
    
    Gate(GateType _gate_type = OTHERGATES)
    {
        gate_name = "";
        gate_type = _gate_type;
    };

    /** const GateType get_type()
     *  ゲートのタイプを返す関数
     */
    const GateType get_type() 
    {
        return (gate_type);
    };

    /** void set_type(GateType _new_type)
     *  ゲートのタイプを設定する関数
     */
    void set_type(GateType _new_type)
    {
        gate_type = _new_type;
    };

    /** int get_count_outputs(void)
     *  ゲートの出力先の数を返す関数
     */
    int get_count_outputs()
    {
        return (count_outputs);
    };
    
    /** void set_count_outputs(int num)
     *  ゲートの出力先の数を設定する関数
     */
    void set_count_outputs(int num)
    {
        count_outputs = num;
    };
    
    /** void set_flag_cuda_d(int f)
     *  ゲートの出力先の数を設定する関数
     */
    void set_flag_cuda_d(int f)
    {
        flag_cuda_d = f;
    };
    
    /** inline void add_input(Gate *g)
     *  void add_input(Gate * g, int i)
     *  入力ゲートgを追加する関数
     */
    void add_input(Gate * g, int i)
    {
        if(input_gates.size() < (unsigned int) (i+1)) {
            input_gates.resize(i+1); //iは０から
        }
        input_gates[i] = g;
    }
    
    inline void add_input(Gate *g)
    {
        int count_in = 0;
        for(int i = 0; i < input_gates.size(); i++){
            if(input_gates[i] -> get_name() == g -> get_name()) count_in = 1;
        }   
        if(count_in == 0) input_gates.push_back(g);
    }

    /** inline void add_output(Gate *g)
     *  出力ゲートgを追加する関数
     */
    inline void add_output(Gate *g)
    {
        int count_out = 0;   
        list<Gate*>::iterator it_out = output_gates.begin();
        
        while( it_out != output_gates.end() )
        {   
            //同じ出力先でも追加するので下の1行をコメントアウト
            // if( (*it_out) -> get_name() == g -> get_name()) count_out = 1;
            ++it_out;
        }
        if(count_out == 0) output_gates.push_back(g);
    }
    
    /** print_in_out()
     *  内容を表示する(自分と、自分の入力ゲートと出力ゲートを出力する)関数
     *  表示例
     *  GateName = xor1
     *  GateType = XOR
     *  InputGate = i0
     *  InputGate = i1
     *  OutputGate = xor3
     *  OutputGate = and4
     */ 
    void print_in_out();
    
    /** \fn void print(std::ostream& os) 
     *  \brief 内容を表示する     
     *  \brief 出力形式: gatename gatetype input1_name input2_name 
     *  \param os [in] 出力ストリーム
     */ 
    void print(ostream& os);      


    /** \fn void print_name(std::ostream& os) 
     *  \brief gateのnameを出力
     *  \param os [in] 出力ストリーム
     */
    void print_name(ostream& os){
      os<< this->gate_name;
    }
    
    /** \fn string get_name()
     *  \brief ゲートの名前を返す
     *  \param
     */
    string get_name(){
        return(gate_name);
    }
    
    /** \fn vector<Gate*> get_input_gate()
     *  \brief あるゲートの入力に繋がったゲートを返す（2入力のゲートなら２つ）
     *  \param
     */
    vector<Gate*> get_input_gate(){
        return(input_gates);
    }
    
     /** \fn int get_output_gates()
     *  \brief ゲートの出力先の数を返す
     *  \param
     */
    int get_output_gates(){
      return(this->output_gates.size());
    }

   /** \fn int get_olevel()
     *  \brief ゲートのoレベルを返す
     *  \param
     */
    int get_olevel(){
      return(this->olevel);
    }

    /** \fn int get_ilevel() 
     *  \brief ゲートのiレベルを返す
     *  \param
     */
    int get_ilevel(){
        return(this->ilevel);
    }

    /** \fn void set_olevel(int level)
     *  \brif 
     *  \param level セットするレベル
     *  
     *  ゲートにレベルを割り当てる
     * 
     */
    void set_olevel(int level) {
      olevel = level;
    }

    /** \fn void set_ilevel(int level)
     *  \brif 
     *  \param level セットするレベル
     *  
     *  ゲートにレベルを割り当てる
     * 
     */
    void set_ilevel(int level) { ilevel = level;}

    /** \fn int calc_olevel()
     *  \brief olevelを計算する
     *  \retval 0 正常に計算できた
     *  \retval 1 正常に計算できなかった
     * 
     *  ゲートのolevelを計算する。直前のゲートのolevelはあらかじめ計算しておく
     *  必要がある。
     */
    int calc_olevel();

    /** \fn int calc_ilevel()
     *  \brief ilevelを計算する
     *  \retval 0 正常に計算できた
     *  \retval 1 正常に計算できなかった
     * 
     *  ゲートのilevelを計算する。直前のゲートのilevelはあらかじめ計算しておく
     *  必要がある。
     */
    int calc_ilevel();

    /** \fn virtual ~Gate()
     *  \brief デストラクタ (今は何もしない）
     */
    virtual ~Gate(){};


    /** \fn int calc_BDDLF()
     *  \brief BDDLFを計算する
     *  \retval 0 正常に計算できた
     *  \retval 1 正常に計算できなかった
     * 
     *  ゲートのBDDLFを計算する。直前のゲートのBDDLFはあらかじめ計算しておく
     *  必要がある。
     */
    int calc_BDDLF();

    /** \fn int calc_TBLF()
     *  \brief TBLFを計算する
     *  \retval 0 正常に計算できた
     *  \retval 1 正常に計算できなかった
     * 
     *  ゲートのTBLFを計算する。直前のゲートのTBLFはあらかじめ計算しておく
     *  必要がある。
     */
    int calc_TBLF(Module* module);
    int calc_TBLF_CUDA(Module* module);
    int calc_TBLF_olev_CUDA(Module* module, int* cash_d, int* input_d);
    int calc_TBLF_ilev_CUDA(Module* module, int* cash_d, int* input_d);

    /** \fn void print_BDDLF(int input)
     *  \brief BDDLF表示する
     *  \param input モジュールの入力ゲート数
     */
    void print_BDDLF(int input);

    /** \fn void print_TBLF(int input)
     *  \brief TBLF表示する
     */
    void print_TBLF(Module* module);

    /** \fn bool is_calc_LF()
     *  \brief LFが計算済みかどうか
     *  \retval true 計算済み
     *  \retval false 未計算
     */
    bool is_calc_LF() {return calc_LF;}

    /** \fn void set_calcLF(bool calcLF)
     *  \brief このゲートのLFの計算済みフラグをセットする
     *  \param calcLF [in] セットするLFの計算済みフラグ
     *
     *  このゲートのLFの計算済みフラグをセットする。
     *
     */
    void set_calc_LF(bool calcLF) { calc_LF = calcLF;}

    /** \fn void setBDDLF(BDD calcBDD)
     *  \brif 
     *  \param calcBDD セットする計算済みBDD
     *  
     *  入力ゲートにBDDを割り当てる
     * 
     */

};


//-----------------------------------------------------------------------------------
//   "Module" class
//-----------------------------------------------------------------------------------

class Module{

private:
    
   /** \var std::string description
    *  \brief モジュール名
    */
   string description;
    
    /** \var std::map<string, Gate*> name2gate
     *  \brief ゲート名からGate*へのハッシュ
     */
    map<string, Gate*> name2gate;

    /** \var list<pair<Gate*, int> > gate_d;
     *  \brief GPUに存在するゲートを管理
     */
    list<pair<Gate*, int> > gate_d;

    /** \var vector<int> gate_d_free;
     *  \brief GPUの空きメモリ（真理値表を入れる用）を管理
     */
    vector<int> gate_d_free;

    /** \var std::map<Gate*, TB*> gate2TB
     *  \brief ゲートからTB*へのハッシュ
     */
    map<Gate*, TB*> gate2TB;

    /** \var std::map<Gate*, int> gate2cash_d
     *  \brief ゲートからgate2cash_d(GPUに存在するTB)へのハッシュ
     */
    map<Gate*, int> gate2cash_d;

   /** \var std::int num_inputs
    *  \brief モジュールの入力ゲートの数
    */
    int num_inputs;

   /** \var std::int max_level
    *  \brief モジュールの最大レベル
    */
    int max_level;

   /** \var std::int threshold_level
    *  \brief レベライズの閾値、threshold_level以降はGPUのメモリに収まる
    */
    int threshold_level;

   /** \var std::vector<Gate*> inputs_
    *  \brief モジュールに含まれる入力端子の配列 [0]から入っている
    */
    vector<Gate*> inputs_;

   /** \var std::list<Gate*> outputs_
    *  \brief モジュールに含まれる出力端子の配列 [0]から入っている
    */
    list<Gate*> outputs_;

   /** \var std::vector<vector<Gate*>> olevel_vector
    *  \brief モジュールに含まれるゲートをoレベル分けし、コンテナのコンテナに格納
    */
    vector< vector<Gate*> > olevel_vector;

   /** \var std::vector<vector<Gate*>> ilevel_vector
    *  \brief モジュールに含まれるゲートをiレベル分けし、コンテナのコンテナに格納
    */
    vector< vector<Gate*> > ilevel_vector;

public:

    list<Gate*> gates_;

    /** \fn string get_word(string &one_line)
   　*  入力ファイルの一行づつを解析していく関数
   　*/
    string get_word(string &one_line);

    /** \fn   Gate * find_name(string _gate_name) 
     *  \brief nameと一致するゲートを返す．なければNULLを返す．
     */
    Gate * find_name(string _gate_name)
    {
        map<string, Gate*>::iterator p;
        p=this->name2gate.find(_gate_name);
        
        if(p != this->name2gate.end()) {
            return p->second;
        } else {
            return NULL;
        }
    }
    
    /** \fn   Gate *set_gate( string _gate_name, string _gate_type )
     *  \brief 送られてきたゲート名のゲートがない場合、新しいゲートを作って登録
     *  \brief 送られてきたゲート名のゲートがある場合、既に登録されているのでtypeのみセットしなおす
     */
    Gate *set_gate( string _gate_name, string _gate_type ){
        Gate * g = this->find_name(_gate_name);        
        if( g == NULL){
            if( _gate_type == "and" ) g = new Gate(_gate_name, AND);
            else if( _gate_type == "or" ) g = new Gate(_gate_name, OR);
            else if( _gate_type == "nand" ) g = new Gate(_gate_name, NAND);
            else if( _gate_type == "nor" ) g = new Gate(_gate_name, NOR);
            else if( _gate_type == "xor" ) g = new Gate(_gate_name, XOR);
            else if( _gate_type == "xnor" ) g = new Gate(_gate_name, XNOR);
            else if( _gate_type == "not" ) g = new Gate(_gate_name, NOT);

            this->insert_name2gate(_gate_name, g);
            this->gates_.push_back(g); 
        }
        else{
            if( _gate_type == "and" ) g -> set_type(AND);
            else if( _gate_type == "or" ) g -> set_type(OR);
            else if( _gate_type == "nand" ) g -> set_type(NAND);
            else if( _gate_type == "nor" ) g -> set_type(NOR);
            else if( _gate_type == "xor" ) g -> set_type(XOR);
            else if( _gate_type == "xnor" ) g -> set_type(XNOR);
            else if( _gate_type == "not" ) g -> set_type(NOT);     
        }
        
        return g;
    }
    
    /** \fn  void  insert_name2gate(string _gate_name, Gate * gate)
     *  \brief name2gateに登録する．既にあるかどうかはチェックしない
     */
    void insert_name2gate(string _gate_name, Gate* gate) {
        this->name2gate.insert(pair<string, Gate*>(_gate_name, gate));
    }

    /** \fn  void insert_gate2cash_d(Gate* gate, int id)
     *  \brief gate2cash_d(GPU用)に登録する．
     */
    void insert_gate2cash_d(Gate* gate, int id) {
        this->gate2cash_d.insert(map<Gate*, int>::value_type(gate, id));
    }
    
    /** \var std::list<Gate*> TBs_
     *  \brief モジュールに含まれるTBリスト
     */
    std::list<TB*> TBs_;

    /** \fn string get_name()
     *  モジュールの名前を返す
     */
    string get_name()
    { 
        return description; 
    };    

    /** \fn std::list<Gate*>& get_gates()
     *  \brief モジュールに含まれるゲートのリストを返す
     *  \brief  ** ONLY for debug usage
     *  \return ゲートのリスト
     */
    std::list<Gate*>& get_gates()
    {
        return gates_;
    };

    /** \fn std::vector<Gate*>& get_inputs()
     *  \brief モジュールに含まれるゲートの入力を返す
     *  \brief  ** ONLY for debug usage
     *  \return ゲートvector
     */
    std::vector<Gate*>& get_inputs()
    {
        return inputs_;
    };

    /** \fn int modulenum_inputs()
     *  \brief モジュールの入力ゲートの数を返す
     *  \return モジュールの入力ゲートの数
     */
    int modulenum_inputs()
    {
        return num_inputs;
    };

    /** \fn int get_numoutputs()
     *  \brief モジュールの出力ゲートの数を返す
     *  \return モジュールの出力ゲートの数
     */
    int get_numoutputs()
    {
        return this->outputs_.size();
    };
   
    /** \fn int get_numgates()
     *  \brief モジュールの全ゲートの数を返す
     *  \return モジュールの全ゲートの数
     */
    int get_numgates()
    {
        return this->gates_.size();
    };

    /** \fn std::list<Gate*>& get_outputs()
     *  \brief モジュールの出力ゲートのlistを返す
     *  \brief  ** ONLY for debug usage
     *  \return ゲートlist
     */
    std::list<Gate*>& get_outputs()
    {
        return outputs_;
    };
    
    /** \fn void Module::read_verilog( char * filename )
     *  \brief 指定されたVerilogファイルを読み込む
     *  \param filename [in] 読み込まれるファイルのファイル名
     * 
     *  filenameで指定されたファイルを読み込む
     */
    void read_verilog( const char * filename );
  
    /** \fn void Module::read_BLIF( char * filename )
     *  \brief 指定されたBLIFファイルを読み込む
     *  \param filename [in] 読み込まれるファイルのファイル名
     * 
     *  filenameで指定されたファイルを読み込む
     */
    void read_BLIF( const char * filename );
  
    /** \fn Module(std::string description = "")
     *  \brief コンストラクタ
     *  \param description [in] モジュール名
     *
     *  モジュール名を指定するコンストラクタ
     */
    Module( string _description = "" )
    {
        description = _description;
    };

    /** \fn virtual ~CircuitPF()
     *  \brief デストラクタ
     *  Moduleのデストラクタ
     */
    virtual ~Module();

    /** \fn void connect(Gate * former_g, Gate * latter_g, int i)
     *  \brief former_gの出力をlatter_gのi番目の入力につなぐ
     *  
     */                          
    void connect(Gate * former_g, Gate * latter_g, int i);

    /** \fn void clear_all_calc_LF() 
     * \brif 全てのLFをクリアする
     *  全てのLFをクリアし、初期化を行う
     */
    void clear_all_calc_LF();

    /** \fn void set_BDD_input();
     * \brif 全ての入力ゲートにBDDを指定する
     */
    void set_BDD_input();

    /** \fn void set_TB_input();
     * \brif 全ての入力ゲートにTBを指定する
     */
    void set_TB_input();

    /** \fn void set_threshold_level(int _threshold)
     * \brif 全ての入力ゲートにTBを指定する
     */
    void set_threshold_level(int _threshold) 
    {
        threshold_level = _threshold;
    }
  
    /** \fn int calc_BDDLF_all()
     *  \brief すべてのゲートのBDDLFを計算する
     *  \retval 0 正常に計算できた
     *  \retval 1 正常に計算できなかった
     *  \see calc_BDDLF()
     *
     *  すべてのゲートのBDDLFを計算する。実際には最後のゲートを指定して
     *  calc_BDDLF() を呼んでいる。また、最後のゲートの出力側のPFの設定も
     *  行う(入力側は行わない)。
     */
    int calc_BDDLF_all();

    /** \fn int calc_TBall()
     *  \brief すべてのゲートのTBを計算する
     *  \retval 0 正常に計算できた
     *  \retval 1 正常に計算できなかった
     *  \see calc_TBLF()
     *
     *  すべてのゲートのTBを計算する。実際には最後のゲートを指定して
     *  calc_TBLF() を呼んでいる。
     */
    int calc_TBall();
    int calc_TBall_CUDA();
    int calc_TBall_olev();
    int calc_TBall_ilev();

    /** \fn   Gate * find_TB(Gate* gate) 
     *  \brief Gateと一致するTBを返す．なければNULLを返す．
     */
    TB * find_TB(Gate* gate)
    {
        std::map<Gate*, TB*>::iterator p;
        p=this->gate2TB.find(gate);
        if(p != this->gate2TB.end()) 
        {
          return p->second;
        } else {
          return NULL;
        }
    }
   
    /** \fn  void insert_gate2TB(Gate* gate, TB * tb)
     *  \brief TBをgate2TBに登録する．既にあるかどうかはチェックしない
     */
    void insert_gate2TB(Gate* gate, TB * tb)
    {
        this->gate2TB.insert(std::pair<Gate*, TB*>(gate, tb));
    }
  
    /** \fn   void simulate_mask_rate()
     *  今回求めるマスク発生率。内容は以下の通り
     *
     *  Simulation(gate set G, input vector set I){
     *  for each i 含む I {
     *      for each g 含む G {   
     *          i の場合のゲートgの出力値を計算;
     *          i の外部出力を計算;
     *          output(g)を反転;
     *          if(2行目の外部出力と5行目で反転した後の外部出力が一致){
     *              DontCare++;
     *          }
     *      }
     *      MaskRate[i] = DontCare/num(G);
     *      DontCare = 0;
     *  }
     *  
     *  for each i 含む I {
     *      AverageMaskRate = sum i 含む I (MaskRate[i])/num(I);
     *  }
     *}
     */
    void simulate_mask_rate();

    /** \fn void set_max_level()
     * \brif モジュール内の最大レベルをセットする。
     */
    void set_max_level(int level)
    {
        max_level = level;
    };

    /** \fn int get_max_level()
     *  モジュールの最大レベルを返す
     */ 
    int get_max_level()
    { 
        return max_level; 
    };

    /** \fn int set_olevel_vector()
     *  \brief すべてのゲートのレベルを初期化する。そして、すべてのゲートにレベルを付ける。
     *  最後に、レベル毎にゲートをコンテナのコンテナ（std::vector<vector<Gate*>> olevel_vector）に仕分けする。 
     *  モジュールの出力ゲートから入力ゲートに向って、それぞれのゲートにレベルをつける。
     *  レベルは、入力側は大きい、出力側は若い。 
     *  レベライズの途中で、元のレベルよりも設定したいレベルが大きい場合は、大きいレベルに更新する。
     *  最後に、モジュール内のmax_levelをセットする。
     *
     *  \retval 0 正常にレベライズできた
     *  \retval 1 正常にレベライズできなかった
     *
     */
    int set_olevel_vector();

    /** \fn int set_ilevel_vector()
     *  \brief すべてのゲートのレベルを初期化する。そして、すべてのゲートにレベルを付ける。
     *  最後に、レベル毎にゲートをコンテナのコンテナ（std::vector<vector<Gate*>> ilevel_vector）に仕分けする。 
     *  モジュールの入力ゲートから出力ゲートに向って、それぞれのゲートにレベルをつける。
     *  レベルは、出力側は大きい、入力側は若い。 
     *  レベライズの途中で、元のレベルよりも設定したいレベルが大きい場合は、大きいレベルに更新する。
     *  最後に、モジュール内のmax_levelをセットする。
     *
     *  \retval 0 正常にレベライズできた
     *  \retval 1 正常にレベライズできなかった
     *
     */
    int set_ilevel_vector();

    /** \fn int find_cash_d(Gate* gate) 
     *  \brief Gateと一致するcash_d(id)を返す．なければ(-1)を返す．
     */
    int find_cash_d(Gate* gate)
    {
        std::map<Gate*, int>::iterator p;
        p = this->gate2cash_d.find(gate);
        if(p != this->gate2cash_d.end()) {
            return (p->second);
        } else {
            return (-1);
        }
    }

    /** \fn int cash_d_size(void) 
     *  \brief gate2cash_dに登録されている数を返す．
     */
    int cash_d_size(void)
    {
        return(this->gate2cash_d.size());
    }

    /** \fn int assign_cash_d(Gate* gate, int* cash_d) 
     *  \brief ゲート（引数）のTBをGPUのメモリに登録(登録のみで転送はしない)
     *  \retval ゲート（引数）のTBのGPU上の番地
     */
    int assign_cash_d(Gate* gate, int* cash_d);

    /** \fn int assign_cash_d_HtoD(Gate* gate, int* cash_d)
     *  \brief ゲート（引数）のホスト側にあるTBをGPUのメモリに登録し、転送する。cudaMemcpyHostToDevice
     *  \retval ゲート（引数）のTBのGPU上の番地
     */
    int assign_cash_d_HtoD(Gate* gate, int* cash_d);

    /** \fn int del_cash_d(Gate* gate) 
     *  \brief ゲート（引数）のTBをGPUのメモリから削除
     */
    int del_cash_d(Gate* gate);
    
    /** \fn int del_cash_d_old(int* cash_d) 
     *  \brief 古いゲートのTBをGPUのメモリから削除
     *   TBをまだ使う場合はswap-outする
     */
    int del_cash_d_old(int* cash_d);

};

#endif //__MULTIGPULOGICSIMULATION_MODULE_HPP