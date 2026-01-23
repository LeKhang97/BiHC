from src.argument import *
from Bio.PDB import MMCIFParser, Structure, Model, Chain

if __name__ == "__main__":
    x, y  = process_args()
    
    if Path(x[0]).exists() == False:
        sys.exit("Filename does not exist!")
    else:
        if y[1]:
            print(f"\nProcessing {x[0]}", end='\n\n')

        with open(x[0], 'r') as infile:
            missing_res = find_missing_res(x[0])

            file = infile.read().split('\n')
            C = process_structure(file, atom_type=y[2], filename=x[0], get_res=True)

            if C == False:
                sys.exit("File is error or chosen atom type is not present!")

            # ===== Get secondary structure (protein-only; CIF -> temp PDB) =====
    chain_types = detect_chain_types(x[0])

    ss_data = get_secondary_structure_stride_auto(x[0], chain_types=chain_types)

    ss_data = get_secondary_structure_stride_auto(x[0], chain_types=chain_types, 
                                                  chain_filter=None if y[5] in ('all', 'longest', None, False) else y[5])

    if '\\' in x[0]:
        filename = ''.join(x[0].split('\\')[-1]).replace('.pdb', '').replace('.cif','')
    else:
        filename = ''.join(x[0].split('/')[-1]).replace('.pdb', '').replace('.cif','')
    
    cmd_file = f'load {os.getcwd()}/{x[0]}; '

    data, res_num_array, removed_chain_index, res_labels_array = check_C(C, x[-1])

    # Get ALL chain IDs (before filtering)
    all_chains = C[1]
    
    # Get remaining chains (after filtering)
    remaining_chains = [all_chains[i] for i in range(len(all_chains)) 
                    if i not in removed_chain_index]

    if data == False:
        sys.exit("File is error!")
    else:
        models = ''
        if bool(models):
            num_chains = len([i for i in remaining_chains if ''.join(i.split('_')[1]) == list(models)[0]])
        else:
            num_chains = len(remaining_chains)
            
        if y[1]:
            print("Number of models (NMR only): ", len(models))
            if models == set(['']):
                print("Model's name has not been specified")
            else:
                if len(models) > 0:
                    print("Models: ", sorted(models), end='\n\n')
            print("Number of chains: ", num_chains, end='\n\n')
        
        if len(data) == 0:
            sys.exit("No chain will be processed!") 

        result = {filename:{}} 
                        
        old_model = ''
        if y[5] == 'longest':
            print(f'Processing longest valid chain!', end='\n\n')
            chains = list(set([i for i in remaining_chains if len(data[remaining_chains.index(i)]) == max([len(data[j]) for j in range(len(data))])]))
        elif y[5] == 'all':
            print(f'Processing all valid chains!', end='\n\n')
            chains = list(set(remaining_chains))
        elif y[5]:
            print(f'Processing chain {y[5]}!', end='\n\n')
            chains = list(set([i for i in remaining_chains if i in [y[5]+'_', y[5]]]))
        else:
            raise ValueError("Cannot find the chain to process, please check your chain name!")

        #print(removed_chain_index, remaining_chains, num_chains, res_num_array, chains) # debug
        
        # ========================================================================
        # FIX CRITICAL: Tạo mapping chain_id -> data index
        # ========================================================================
        # Build mapping: chain_id -> index in data/res_num_array/res_labels_array
        chain_to_data_idx = {chain_id: idx for idx, chain_id in enumerate(remaining_chains)}
        
        for chain_id in chains:
            data_idx = chain_to_data_idx[chain_id]
            
            subdata = data[data_idx]
            res_num = res_num_array[data_idx]
            current_res_labels = res_labels_array[data_idx]
            name = filename + f'_chain_{chain_id}'
            
            # Debug
            if y[1]:
                print(f"\nProcessing chain: {chain_id}")
                print(f"  Data index: {data_idx}")
                print(f"  Residue range: {current_res_labels[0]} - {current_res_labels[-1]}")
                print(f"  Number of residues: {len(current_res_labels)}")
            
            if 'MODEL' in chain_id:
                chain = chain_id.split('_')[0]
                model = chain_id.split('_')[1]
            else:
                model = ''
                chain = chain_id.replace('_','')
            
            if model != old_model:
                old_model = model

            name = filename + f'_chain_{chain}'

            if 'ss_by_chain' not in locals():
                ss_by_chain = {}

            # only for protein chains that exist in ss_data
            if chain_types.get(chain) == 'protein' and chain in ss_data:
                # map res_label -> ss
                ss_map = {str(r): s for (r, s) in ss_data[chain]}

                # align SSE to current_res_labels order (best effort)
                ss_aligned = [ss_map.get(str(r), 'C') for r in current_res_labels]

                # also ensure length matches subdata
                if len(ss_aligned) != len(subdata):
                    ss_aligned = ss_aligned[:len(subdata)] + ['C'] * max(0, len(subdata) - len(ss_aligned))

                ss_by_chain[chain_id] = ss_aligned
            else:
                # no SSE -> leave missing so B2 will fallback to B
                pass

            set_ss_by_chain(ss_by_chain)  # BiHC
            
            pred = cluster_algo(subdata, *x[1:], chain, chain_types.get(chain))

            #subdata, algo, top_thres, bot_thres, resol, weight, dist_thres, len_thres, chain, chain_type)

            # ===== Curate by secondary structure =====
            '''if chain_types.get(chain) == 'protein' and chain in ss_data:
                print("Refining clustering based on secondary structure...")
                pred = curate_by_secondary_structure(
                    pred, 
                    current_res_labels, 
                    ss_data[chain],
                    min_ss_length=3
                )'''
            #print('ss data: ', bool(ss_data))

            '''if y[7]:
                print('Post-processing clustering result...')
                l1, l2, l3 = [int(j) for j in y[8].replace(' ','').split(';')]
                pred = post_process_update(pred, res_num, segment_length = [l1,l2,l3])'''
            
            num_clusters = len(set(i for i in pred if i != -1))
            #outlier = 'with' if -1 in pred else 'without'

            msg = 'Output information:'
            decorate_message(msg)
            #print(f'Chain {chain} has {num_clusters} clusters and {outlier} outliers.')

            pred_ext, res_num_ext, res_labels_ext = extend_missing_res(
                pred, res_num, res_labels=current_res_labels
            )

            use_pred = pred_ext
            use_res_num = res_num_ext
            use_res_labels = res_labels_ext

            for h, k in enumerate(set(j for j in use_pred if j != -1)):
                positions = [use_res_labels[j] for j in range(len(use_pred)) if use_pred[j] == k]
                range_pos = list_to_range_with_insertions(positions)
                mess_pos = format_range_string(range_pos)
                
                print(f'Number of residues of cluster {h+1}: {len([j for j in use_pred if j == k])}')
                print(f'Cluster {h+1} positions:\n{mess_pos}\n')
            
            if -1 in use_pred:
                positions = [use_res_labels[j] for j in range(len(use_pred)) if use_pred[j] == -1]
                range_pos = list_to_range_with_insertions(positions)
                mess_pos = format_range_string(range_pos)
                
                print(f'Number of residues of outliers: {len([j for j in use_pred if j == -1])}')
                #print(f'Outliers positions:\n{mess_pos}\n')

            pymol_cmd = pymol_process(
                use_pred, 
                use_res_num, 
                name, 
                verbose=y[1], 
                res_labels=use_res_labels
            )

            print('\n')
            
            result[filename][chain_id.replace('_','')] = {
                'data': subdata,
                'cluster': pred,
                'res': res_num,
                'res_labels': current_res_labels,
                'missing_res': missing_res[chain_id] if chain_id in missing_res else None,
                'PyMOL': pymol_cmd
            }
            cmd_file += '; '.join(pymol_cmd) + ';'
            
            if y[6]:
                ref = process_ref_txt(y[6], res_num)
                result[filename][chain_id.replace('_','')]['cluster_ref'] = ref

                lsts_label = [ref, pred]
                print(lsts_label[1])
                db = domain_distance_matrix(lsts_label, res_num)
                db2 = domain_distance_matrix2(lsts_label, res_num)
                try:
                    csd = round(CSD(db2, threshold=20), 3)
                    dbd = round(DBD(db, threshold=20), 3)
                except:
                    raise
                
                dom, min_labels = domain_overlap_matrix(lsts_label, res_num)
                ndo = round(NDO(dom, len(res_num), min_labels), 3)
                iou = round(IoU(dom, [ref, pred]), 3)
                print(f"CSD: {csd}, IoU: {iou}, NDO: {ndo}, DBD: {dbd}")

                result[filename][chain_id.replace('_','')]['Score'] = {
                    "CSD": csd, "IoU": iou, "NDO": ndo, "DBD": dbd
                }
            else:
                result[filename][chain_id.replace('_','')]['cluster_ref'] = None
                
    if y[0] != None:
        target_dir = Path(y[0])
        if y[1]:
            msg = 'Exporting output:'
            decorate_message(msg)
            print(f"Writing to the path {target_dir}", end='\n\n')
        
        target_dir.mkdir(parents=True, exist_ok=True)

        if y[3] != None:
            basename1 = os.path.basename(y[3])

            outfile1 = target_dir / f"{basename1.replace('.json','').replace('.pdb','').replace('.cif','')}.json"
            outfile2 = target_dir / f"{basename1.replace('.json','').replace('.pdb','').replace('.cif','')}_pymolcmd.pml"
            
            with open(outfile1, 'w') as outfile:
                json.dump(result, outfile, indent=2, cls=CustomNumpyEncoder)
        
            with open(outfile2, 'w') as outfile:
                outfile.write(cmd_file)
            
            if y[1]:
                print(f"Wrote {outfile1} and {outfile2}")

        if y[4] != None:
            basename2 = os.path.basename(y[4])
            name = x[0].replace('.pdb', '').replace('.cif', '')
            is_cif_input = x[0].lower().endswith('.cif')
            
            for chain in result[filename].keys():
                pred = result[filename][chain]['cluster']
                res_num = result[filename][chain]['res']
                res = list(res_num) 

                cluster_result = process_cluster_format(pred, res)
                cluster_lines, is_cif = split_pdb_by_clusters(x[0], cluster_result, name, chain)
                
                if y[1]:
                    print(f"Writing clusters for chain {chain}...")
                
                file_ext = '.cif' if is_cif_input else '.pdb'
                output_file = target_dir / f"{filename}{file_ext}"

                with open(output_file, 'w') as outfile:
                    if is_cif:
                        try:
                            from Bio.PDB import MMCIFIO
                        except ModuleNotFoundError:
                            print("Warning: Biopython not available, skipping CIF output")
                            continue
                        
                        parser = MMCIFParser(QUIET=True)
                        original_structure = parser.get_structure('original', x[0])
                        
                        new_structure = Structure.Structure('clustered')
                        new_model = Model.Model(0)
                        new_structure.add(new_model)
                        
                        all_cluster_data = [item for sublist in cluster_lines.values() for item in sublist]
                        
                        chain_dict = {}
                        for chain_id, residue, atom in all_cluster_data:
                            if chain_id not in chain_dict:
                                chain_dict[chain_id] = {}
                            res_id = residue.id
                            if res_id not in chain_dict[chain_id]:
                                chain_dict[chain_id][res_id] = residue
                        
                        for chain_id in sorted(chain_dict.keys()):
                            new_chain = Chain.Chain(chain_id)
                            for res_id in sorted(chain_dict[chain_id].keys()):
                                new_chain.add(chain_dict[chain_id][res_id].copy())
                            new_model.add(new_chain)
                        
                        io = MMCIFIO()
                        io.set_structure(new_structure)
                        io.save(str(output_file))
                    else:
                        outfile.writelines([i for j in cluster_lines.keys() for i in cluster_lines[j]])
        
            if y[1]:
                print("Writing completed!")